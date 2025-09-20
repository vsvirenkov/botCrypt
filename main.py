import asyncio
import json
import logging
import time
from datetime import datetime
import telegram
from pybit.unified_trading import HTTP

from dotenv import load_dotenv
import os
load_dotenv()  # Загружает .env

# Настройка логирования
logging.basicConfig(
    filename="bybit_funding_rate_bot.log",
    format="%(asctime)s %(message)s",
    level=logging.INFO
)

# Конфигурация
SYMBOL = "DOGEUSDT"  # Торговая пара для спота и перпетуала (альтернатива: "BTCUSDT")
STABLE = "USDT"     # Стабильная монета
POSITION_SIZE = 5  # Сумма в USDT для каждой позиции (увеличено для мин. суммы)
CHECK_INTERVAL = 3600  # Проверка funding rate каждые 3600 секунд (1 час)
FUNDING_RATE_THRESHOLD = 0.01  # Минимальный funding rate для входа (%)
ORDER_TYPE = "Market"  # Тип ордера: "Market" или "Limit"

# Telegram настройки
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Загрузка API ключей из файла
API_KEY = os.getenv("API_KEY_BYBIT")
API_SECRET = os.getenv("API_SECRET_BYBIT")

# Инициализация клиентов
session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=os.getenv("TEST"))  # testnet=True для тестовой среды, False для реальной
bot = telegram.Bot(token=TELEGRAM_TOKEN)

async def send_telegram_message(message):
    """Отправка уведомления в Telegram"""
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message)
        logging.info(f"Telegram уведомление отправлено: {message}")
    except Exception as e:
        logging.error(f"Ошибка Telegram: {e}")

def get_instrument_info(category, symbol):
    """Получение информации о торговой паре (минимальный размер, точность, мин. сумма)"""
    try:
        response = session.get_instruments_info(category=category, symbol=symbol)
        if response.get("retCode") != 0:
            logging.error(f"Ошибка получения информации о паре: {response.get('retMsg')}")
            return None
        instrument = response["result"]["list"][0]
        lot_size_filter = instrument.get("lotSizeFilter", {})
        min_order_qty = float(lot_size_filter.get("minOrderQty", "0.001"))
        qty_precision = len(lot_size_filter.get("qtyStep", "0.0001").split(".")[-1])
        min_order_amt = float(lot_size_filter.get("minOrderAmt", "100")) if category == "spot" else 0.0
        logging.info(f"Информация о паре {category} {symbol}: minOrderQty={min_order_qty}, qtyPrecision={qty_precision}, minOrderAmt={min_order_amt}")
        return {
            "minOrderQty": min_order_qty,
            "qtyPrecision": qty_precision,
            "minOrderAmt": min_order_amt
        }
    except Exception as e:
        logging.error(f"Ошибка получения информации о паре: {e}")
        return None

def get_funding_rate(symbol):
    """Получение текущего funding rate через get_tickers"""
    try:
        response = session.get_tickers(category="linear", symbol=symbol)
        if response.get("retCode") != 0:
            logging.error(f"Ошибка API: {response.get('retMsg')}, Параметры: category=linear, symbol={symbol}")
            return None
        funding_rate = float(response["result"]["list"][0]["fundingRate"])
        return funding_rate * 100  # Конвертация в проценты
    except Exception as e:
        logging.error(f"Ошибка получения funding rate: {e}")
        return None

def get_spot_price(symbol):
    """Получение текущей цены на споте"""
    try:
        ticker = session.get_tickers(category="spot", symbol=symbol)
        return float(ticker["result"]["list"][0]["lastPrice"])
    except Exception as e:
        logging.error(f"Ошибка получения спотовой цены: {e}")
        return None

def get_perp_price(symbol):
    """Получение текущей цены на перпетуале"""
    try:
        ticker = session.get_tickers(category="linear", symbol=symbol)
        return float(ticker["result"]["list"][0]["lastPrice"])
    except Exception as e:
        logging.error(f"Ошибка получения цены перпетуала: {e}")
        return None

def get_available_balance(coin):
    """Проверка доступного баланса"""
    try:
        balance = session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        coin_list = balance["result"]["list"][0]["coin"]
        for c in coin_list:
            if c["coin"] == coin:
                balance_value = c.get("walletBalance", c.get("availableToWithdraw", "0"))
                return float(balance_value)
        logging.error(f"Монета {coin} не найдена в балансе")
        return 0.0
    except Exception as e:
        logging.error(f"Ошибка получения баланса: {e}, Ответ: {balance}")
        return None

def place_spot_order(symbol, side, qty, min_order_qty, qty_precision, min_order_amt, spot_price):
    """Размещение спотового ордера"""
    try:
        qty = max(round(qty, qty_precision), min_order_qty)  # Учитываем минимальный размер
        order_value = qty * spot_price
        if qty < min_order_qty:
            logging.error(f"Количество {qty} меньше минимального {min_order_qty} для спотового ордера")
            return None
        if order_value < min_order_amt:
            logging.error(f"Стоимость ордера {order_value} USDT меньше минимальной {min_order_amt} USDT")
            return None
        logging.info(f"Размещение спотового ордера: qty={qty}, value={order_value} USDT")
        order_params = {
            "category": "spot",
            "symbol": symbol,
            "side": side,
            "orderType": ORDER_TYPE,
            "qty": str(qty)
        }
        if ORDER_TYPE == "Limit":
            order_params["price"] = str(spot_price)
        response = session.place_order(**order_params)
        if response.get("retCode") != 0:
            error_msg = f"Ошибка API при размещении спотового ордера ({ORDER_TYPE}): {response.get('retMsg')}, Ответ: {response}"
            logging.error(error_msg)
            asyncio.create_task(send_telegram_message(error_msg))
            return None
        return response["result"]["orderId"]
    except Exception as e:
        logging.error(f"Ошибка размещения спотового ордера: {e}")
        return None

def place_perp_order(symbol, side, qty, min_order_qty, qty_precision, perp_price):
    """Размещение перпетуального ордера"""
    try:
        qty = max(round(qty, qty_precision), min_order_qty)  # Учитываем минимальный размер
        if qty < min_order_qty:
            logging.error(f"Количество {qty} меньше минимального {min_order_qty} для перпетуального ордера")
            return None
        logging.info(f"Размещение перпетуального ордера: qty={qty}")
        order_params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": ORDER_TYPE,
            "qty": str(qty)
        }
        if ORDER_TYPE == "Limit":
            order_params["price"] = str(perp_price)
        response = session.place_order(**order_params)
        if response.get("retCode") != 0:
            error_msg = f"Ошибка API при размещении перпетуального ордера ({ORDER_TYPE}): {response.get('retMsg')}, Ответ: {response}"
            logging.error(error_msg)
            asyncio.create_task(send_telegram_message(error_msg))
            return None
        return response["result"]["orderId"]
    except Exception as e:
        logging.error(f"Ошибка размещения перпетуального ордера: {e}")
        return None

async def funding_rate_arbitrage():
    """Основная логика арбитража funding rate"""
    while True:
        try:
            # Проверка баланса
            available = get_available_balance(STABLE)
            if available is None:
                message = "Ошибка: не удалось получить баланс"
                await send_telegram_message(message)
                await asyncio.sleep(CHECK_INTERVAL)
                continue
            if available < POSITION_SIZE * 2:
                message = f"Недостаточный баланс: {available} {STABLE}. Требуется: {POSITION_SIZE * 2}"
                await send_telegram_message(message)
                logging.info(message)
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            # Проверка funding rate
            funding_rate = get_funding_rate(SYMBOL)
            if funding_rate is None:
                message = "Ошибка: не удалось получить funding rate"
                await send_telegram_message(message)
                await asyncio.sleep(CHECK_INTERVAL)
                continue
            if funding_rate < FUNDING_RATE_THRESHOLD:
                message = f"Funding rate {funding_rate}% ниже порога {FUNDING_RATE_THRESHOLD}%"
                await send_telegram_message(message)
                logging.info(message)
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            # Получение цен
            spot_price = get_spot_price(SYMBOL)
            perp_price = get_perp_price(SYMBOL)
            if spot_price is None or perp_price is None:
                message = "Ошибка: не удалось получить цены"
                await send_telegram_message(message)
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            # Получение информации о минимальном размере и точности
            spot_info = get_instrument_info("spot", SYMBOL)
            perp_info = get_instrument_info("linear", SYMBOL)
            if not spot_info or not perp_info:
                message = f"Ошибка: не удалось получить информацию о торговой паре {SYMBOL}"
                await send_telegram_message(message)
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            qty = POSITION_SIZE / spot_price  # Количество монет для позиции
            logging.info(f"Рассчитанное qty: {qty}, spot_price: {spot_price}")

            # Размещение ордеров
            spot_order_id = place_spot_order(SYMBOL, "Buy", qty, spot_info["minOrderQty"], spot_info["qtyPrecision"], spot_info["minOrderAmt"], spot_price)
            perp_order_id = place_perp_order(SYMBOL, "Sell", qty, perp_info["minOrderQty"], perp_info["qtyPrecision"], perp_price)

            if spot_order_id and perp_order_id:
                message = (f"Открыта арбитражная позиция:\n"
                          f"Спот: Куплено {qty:.6f} {SYMBOL} по {spot_price}\n"
                          f"Перпетуал: Продано {qty:.6f} {SYMBOL} по {perp_price}\n"
                          f"Funding Rate: {funding_rate}%")
            else:
                message = f"Ошибка: не удалось открыть одну или обе позиции для {SYMBOL}"
            await send_telegram_message(message)
            logging.info(message)

        except Exception as e:
            message = f"Общая ошибка: {e}"
            await send_telegram_message(message)
            logging.error(message)

        await asyncio.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    asyncio.run(funding_rate_arbitrage())