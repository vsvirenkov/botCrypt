import asyncio
import json
import logging
import time
import os
from datetime import datetime, timedelta
import telegram
from pybit.unified_trading import HTTP
from typing import Dict, List, Optional, Tuple
import signal
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'bybit_funding_bot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BybitFundingBot:
    def __init__(self):
        # Конфигурация
        self.SYMBOLS = ["DOGEUSDT", "ETHUSDT"]  # Поддерживаемые пары
        self.STABLE = "USDT"
        self.POSITION_SIZE = 5  # USDT на каждую позицию
        self.CHECK_INTERVAL = 1800  # 30 минут
        self.FUNDING_RATE_THRESHOLD = 0.02  # 0.02% минимальный funding rate
        self.MAX_POSITIONS_PER_SYMBOL = 1  # Максимум позиций на пару
        self.ORDER_TYPE = "Market"  # "Market" или "Limit"
        self.STOP_LOSS_PERCENT = 0.05  # 5% стоп-лосс
        self.CLOSE_NEGATIVE_RATE = True  # Закрывать при отрицательном funding rate

        # Telegram настройки
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
        self.CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

        # API ключи из переменных окружения (более безопасно)
        self.API_KEY = os.getenv("BYBIT_API_KEY")
        self.API_SECRET = os.getenv("BYBIT_API_SECRET")

        # Проверка обязательных переменных
        if not all([self.API_KEY, self.API_SECRET, self.TELEGRAM_TOKEN, self.CHAT_ID]):
            raise ValueError("Необходимые переменные окружения не установлены!")

        # Инициализация клиентов
        self.session = HTTP(
            api_key=self.API_KEY,
            api_secret=self.API_SECRET,
            testnet=False  # Для продакшн
        )
        self.bot = telegram.Bot(token=self.TELEGRAM_TOKEN)

        # Состояние бота
        self.active_positions = {}  # {symbol: {'spot_order_id': str, 'perp_order_id': str, 'open_time': datetime}}
        self.running = True

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"🚀 Bybit Funding Rate Bot запущен для пар: {self.SYMBOLS}")
        logger.info(f"💰 Размер позиции: {self.POSITION_SIZE} USDT")
        logger.info(f"📊 Порог funding rate: {self.FUNDING_RATE_THRESHOLD}%")

    def _signal_handler(self, signum, frame):
        """Обработка сигналов для graceful shutdown"""
        logger.info(f"Получен сигнал {signum}. Завершение работы...")
        self.running = False
        asyncio.create_task(self._cleanup())

    async def _cleanup(self):
        """Очистка при завершении работы"""
        logger.info("🔄 Закрытие всех активных позиций...")
        for symbol in list(self.active_positions.keys()):
            await self._close_position(symbol)
        logger.info("✅ Бот остановлен")
        sys.exit(0)

    async def send_telegram_message(self, message: str, parse_mode: str = None):
        """Отправка уведомления в Telegram"""
        try:
            await self.bot.send_message(
                chat_id=self.CHAT_ID,
                text=message,
                parse_mode=parse_mode
            )
            logger.info(f"📱 Telegram: {message[:100]}...")
        except Exception as e:
            logger.error(f"❌ Ошибка Telegram: {e}")

    def get_instrument_info(self, category: str, symbol: str) -> Optional[Dict]:
        """Получение информации о торговой паре"""
        try:
            response = self.session.get_instruments_info(category=category, symbol=symbol)
            if response.get("retCode") != 0:
                logger.error(f"❌ Ошибка получения информации о {symbol}: {response.get('retMsg')}")
                return None

            instrument = response["result"]["list"][0]
            lot_size_filter = instrument.get("lotSizeFilter", {})
            min_order_qty = float(lot_size_filter.get("minOrderQty", "0.001"))
            qty_precision = len(lot_size_filter.get("qtyStep", "0.0001").split(".")[-1])
            min_order_amt = float(lot_size_filter.get("minOrderAmt", "100")) if category == "spot" else 0.0

            logger.info(f"📊 {category.upper()} {symbol}: minQty={min_order_qty}, precision={qty_precision}, minAmt={min_order_amt}")
            return {
                "minOrderQty": min_order_qty,
                "qtyPrecision": qty_precision,
                "minOrderAmt": min_order_amt
            }
        except Exception as e:
            logger.error(f"❌ Ошибка получения информации о {symbol}: {e}")
            return None

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Получение текущего funding rate"""
        try:
            response = self.session.get_tickers(category="linear", symbol=symbol)
            if response.get("retCode") != 0:
                logger.error(f"❌ Ошибка получения funding rate для {symbol}: {response.get('retMsg')}")
                return None

            funding_rate = float(response["result"]["list"][0]["fundingRate"])
            rate_percent = funding_rate * 100
            logger.info(f"💹 {symbol}: Funding Rate {rate_percent:.4f}%")
            return rate_percent
        except Exception as e:
            logger.error(f"❌ Ошибка получения funding rate для {symbol}: {e}")
            return None

    def get_spot_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены на споте"""
        try:
            ticker = self.session.get_tickers(category="spot", symbol=symbol)
            return float(ticker["result"]["list"][0]["lastPrice"])
        except Exception as e:
            logger.error(f"❌ Ошибка получения спотовой цены {symbol}: {e}")
            return None

    def get_perp_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены на перпетуале"""
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            return float(ticker["result"]["list"][0]["lastPrice"])
        except Exception as e:
            logger.error(f"❌ Ошибка получения цены перпетуала {symbol}: {e}")
            return None

    def get_available_balance(self, coin: str) -> Optional[float]:
        """Проверка доступного баланса"""
        try:
            balance = self.session.get_wallet_balance(accountType="UNIFIED", coin=coin)
            coin_list = balance["result"]["list"][0]["coin"]
            for c in coin_list:
                if c["coin"] == coin:
                    balance_value = c.get("walletBalance", c.get("availableToWithdraw", "0"))
                    return float(balance_value)
            logger.warning(f"⚠️ Монета {coin} не найдена в балансе")
            return 0.0
        except Exception as e:
            logger.error(f"❌ Ошибка получения баланса {coin}: {e}")
            return None

    def calculate_qty(self, position_size: float, price: float, min_order_qty: float, qty_precision: int) -> float:
        """Расчет количества с учетом ограничений"""
        qty = position_size / price
        qty = max(round(qty, qty_precision), min_order_qty)
        return qty

    async def place_spot_order(self, symbol: str, side: str, qty: float,
                              min_order_qty: float, qty_precision: int,
                              min_order_amt: float, spot_price: float) -> Optional[str]:
        """Размещение спотового ордера"""
        try:
            order_value = qty * spot_price
            if qty < min_order_qty:
                logger.error(f"❌ {symbol} Spot: qty {qty} < min {min_order_qty}")
                return None
            if order_value < min_order_amt:
                logger.error(f"❌ {symbol} Spot: value {order_value} < min {min_order_amt}")
                return None

            order_params = {
                "category": "spot",
                "symbol": symbol,
                "side": side,
                "orderType": self.ORDER_TYPE,
                "qty": str(qty)
            }
            if self.ORDER_TYPE == "Limit":
                order_params["price"] = str(spot_price)

            logger.info(f"📈 {symbol} Spot {side}: qty={qty}, value={order_value:.2f} USDT")
            response = self.session.place_order(**order_params)

            if response.get("retCode") != 0:
                error_msg = f"❌ {symbol} Spot ошибка: {response.get('retMsg')} (Code: {response.get('retCode')})"
                logger.error(error_msg)
                await self.send_telegram_message(error_msg)
                return None

            order_id = response["result"]["orderId"]
            logger.info(f"✅ {symbol} Spot ордер размещен: {order_id}")
            return order_id

        except Exception as e:
            logger.error(f"❌ {symbol} Spot исключение: {e}")
            return None

    async def place_perp_order(self, symbol: str, side: str, qty: float,
                              min_order_qty: float, qty_precision: int,
                              perp_price: float) -> Optional[str]:
        """Размещение перпетуального ордера"""
        try:
            if qty < min_order_qty:
                logger.error(f"❌ {symbol} Perp: qty {qty} < min {min_order_qty}")
                return None

            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": self.ORDER_TYPE,
                "qty": str(qty)
            }
            if self.ORDER_TYPE == "Limit":
                order_params["price"] = str(perp_price)

            logger.info(f"📉 {symbol} Perp {side}: qty={qty}")
            response = self.session.place_order(**order_params)

            if response.get("retCode") != 0:
                error_msg = f"❌ {symbol} Perp ошибка: {response.get('retMsg')} (Code: {response.get('retCode')})"
                logger.error(error_msg)
                await self.send_telegram_message(error_msg)
                return None

            order_id = response["result"]["orderId"]
            logger.info(f"✅ {symbol} Perp ордер размещен: {order_id}")

            # Добавляем стоп-лосс для перпетуальной позиции
            if self.STOP_LOSS_PERCENT > 0:
                await self._set_perp_stop_loss(symbol, side, qty, perp_price)

            return order_id

        except Exception as e:
            logger.error(f"❌ {symbol} Perp исключение: {e}")
            return None

    async def _set_perp_stop_loss(self, symbol: str, side: str, qty: float, entry_price: float):
        """Установка стоп-лосса для перпетуальной позиции"""
        try:
            if side == "Sell":  # Short позиция
                stop_price = entry_price * (1 + self.STOP_LOSS_PERCENT)
                stop_side = "Buy"  # Закрытие short
            else:  # Long позиция (если добавим)
                stop_price = entry_price * (1 - self.STOP_LOSS_PERCENT)
                stop_side = "Sell"

            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=stop_side,
                orderType="Market",
                qty=str(qty),
                stopLoss=str(stop_price),
                triggerPrice=str(stop_price),
                triggerBy="LastPrice",
                orderLinkId=f"{symbol}_stoploss_{int(time.time())}"
            )

            if response.get("retCode") == 0:
                logger.info(f"🛡️ {symbol} Stop-loss установлен: {stop_price}")
            else:
                logger.warning(f"⚠️ {symbol} Не удалось установить stop-loss: {response.get('retMsg')}")

        except Exception as e:
            logger.error(f"❌ {symbol} Ошибка stop-loss: {e}")

    async def _close_position(self, symbol: str) -> bool:
        """Закрытие арбитражной позиции"""
        try:
            if symbol not in self.active_positions:
                logger.info(f"ℹ️ {symbol}: Позиция не найдена для закрытия")
                return True

            position = self.active_positions[symbol]
            close_time = datetime.now()
            duration = (close_time - position["open_time"]).total_seconds() / 3600  # в часах

            # Закрытие спотовой позиции
            if "spot_order_id" in position:
                spot_response = self.session.cancel_order(
                    category="spot",
                    symbol=symbol,
                    orderId=position["spot_order_id"]
                )
                logger.info(f"🔄 {symbol} Spot закрытие: {spot_response.get('retMsg', 'OK')}")

            # Закрытие перпетуальной позиции
            if "perp_order_id" in position:
                perp_response = self.session.close_position(
                    category="linear",
                    symbol=symbol
                )
                logger.info(f"🔄 {symbol} Perp закрытие: {perp_response.get('retMsg', 'OK')}")

            # Удаление из активных позиций
            del self.active_positions[symbol]

            message = f"🔒 {symbol} позиция закрыта\n⏱️ Время работы: {duration:.1f}ч"
            await self.send_telegram_message(message)
            logger.info(message)

            return True

        except Exception as e:
            logger.error(f"❌ {symbol} Ошибка закрытия позиции: {e}")
            return False

    async def check_existing_positions(self, symbol: str) -> bool:
        """Проверка существующих позиций"""
        try:
            # Проверка перпетуальной позиции
            positions = self.session.get_positions(category="linear", symbol=symbol)
            for pos in positions["result"]["list"]:
                if float(pos["size"]) > 0:
                    logger.info(f"ℹ️ {symbol}: Существующая позиция обнаружена, пропускаем")
                    return True

            return False
        except Exception as e:
            logger.error(f"❌ {symbol} Ошибка проверки позиций: {e}")
            return False

    async def open_arbitrage_position(self, symbol: str) -> bool:
        """Открытие арбитражной позиции"""
        try:
            # Проверка существующих позиций
            if await self.check_existing_positions(symbol):
                return False

            # Получение цен
            spot_price = self.get_spot_price(symbol)
            perp_price = self.get_perp_price(symbol)
            if not spot_price or not perp_price:
                logger.error(f"❌ {symbol}: Не удалось получить цены")
                return False

            # Получение информации о паре
            spot_info = self.get_instrument_info("spot", symbol)
            perp_info = self.get_instrument_info("linear", symbol)
            if not spot_info or not perp_info:
                logger.error(f"❌ {symbol}: Не удалось получить информацию о паре")
                return False

            # Расчет количества
            qty = self.calculate_qty(
                self.POSITION_SIZE, spot_price,
                spot_info["minOrderQty"], spot_info["qtyPrecision"]
            )

            # Проверка баланса
            available = self.get_available_balance(self.STABLE)
            if available is None or available < self.POSITION_SIZE * 2:
                logger.warning(f"⚠️ {symbol}: Недостаточно баланса {available}")
                return False

            # Размещение ордеров
            spot_order_id = await self.place_spot_order(
                symbol, "Buy", qty, spot_info["minOrderQty"],
                spot_info["qtyPrecision"], spot_info["minOrderAmt"], spot_price
            )

            if not spot_order_id:
                logger.error(f"❌ {symbol}: Не удалось разместить спотовый ордер")
                return False

            perp_order_id = await self.place_perp_order(
                symbol, "Sell", qty, perp_info["minOrderQty"],
                perp_info["qtyPrecision"], perp_price
            )

            if not perp_order_id:
                logger.error(f"❌ {symbol}: Не удалось разместить перпетуальный ордер")
                # Откатываем спотовый ордер
                await self.session.cancel_order(category="spot", symbol=symbol, orderId=spot_order_id)
                return False

            # Сохранение активной позиции
            self.active_positions[symbol] = {
                "spot_order_id": spot_order_id,
                "perp_order_id": perp_order_id,
                "open_time": datetime.now(),
                "spot_price": spot_price,
                "perp_price": perp_price,
                "qty": qty
            }

            message = (
                f"🚀 {symbol} Арбитражная позиция открыта!\n"
                f"📈 Спот: Куплено {qty:.6f} по {spot_price:,.2f}\n"
                f"📉 Перп: Продано {qty:.6f} по {perp_price:,.2f}\n"
                f"💰 Размер: {POSITION_SIZE} USDT\n"
                f"🔒 Stop-loss: {self.STOP_LOSS_PERCENT*100}%"
            )
            await self.send_telegram_message(message, parse_mode="HTML")
            logger.info(f"✅ {symbol}: Арбитражная позиция открыта")

            return True

        except Exception as e:
            logger.error(f"❌ {symbol}: Ошибка открытия позиции: {e}")
            return False

    async def monitor_positions(self):
        """Мониторинг активных позиций"""
        try:
            for symbol in list(self.active_positions.keys()):
                position = self.active_positions[symbol]
                open_time = position["open_time"]
                duration = (datetime.now() - open_time).total_seconds() / 3600  # в часах

                # Проверка funding rate для закрытия
                if self.CLOSE_NEGATIVE_RATE:
                    funding_rate = self.get_funding_rate(symbol)
                    if funding_rate is not None and funding_rate < 0:
                        logger.info(f"📉 {symbol}: Funding rate отрицательный ({funding_rate:.2f}%)")
                        await self._close_position(symbol)
                        continue

                # Периодический статус
                if duration % 8 < 0.1:  # Каждые 8 часов
                    funding_rate = self.get_funding_rate(symbol)
                    if funding_rate:
                        message = f"📊 {symbol} статус\n⏱️ Время: {duration:.1f}ч\n💹 Funding: {funding_rate:.4f}%"
                        await self.send_telegram_message(message)

        except Exception as e:
            logger.error(f"❌ Ошибка мониторинга позиций: {e}")

    async def check_balance_alert(self):
        """Проверка баланса и алерты"""
        try:
            available = self.get_available_balance(self.STABLE)
            if available is None:
                return

            total_required = len(self.SYMBOLS) * self.POSITION_SIZE * 2
            if available < total_required * 0.5:  # Менее 50% от требуемого
                message = f"⚠️ Низкий баланс!\n💰 Доступно: {available:.2f} {self.STABLE}\n📊 Требуется: {total_required:.2f} {self.STABLE}"
                await self.send_telegram_message(message)
                logger.warning(message)

        except Exception as e:
            logger.error(f"❌ Ошибка проверки баланса: {e}")

    async def main_loop(self):
        """Основной цикл бота"""
        logger.info("🔄 Запуск основного цикла...")

        while self.running:
            try:
                # Проверка баланса
                await self.check_balance_alert()

                # Мониторинг позиций
                await self.monitor_positions()

                # Проверка новых возможностей
                for symbol in self.SYMBOLS:
                    if symbol in self.active_positions and len(self.active_positions[symbol]) >= self.MAX_POSITIONS_PER_SYMBOL:
                        continue

                    funding_rate = self.get_funding_rate(symbol)
                    if funding_rate is None:
                        continue

                    if funding_rate > self.FUNDING_RATE_THRESHOLD:
                        logger.info(f"🎯 {symbol}: Funding rate {funding_rate:.4f}% > порога {self.FUNDING_RATE_THRESHOLD}%")
                        success = await self.open_arbitrage_position(symbol)
                        if success:
                            await asyncio.sleep(60)  # Пауза после открытия позиции
                    else:
                        if funding_rate < 0 and symbol in self.active_positions:
                            logger.info(f"📉 {symbol}: Funding rate отрицательный, закрываем")
                            await self._close_position(symbol)

                # Пауза между циклами
                logger.debug(f"😴 Пауза {self.CHECK_INTERVAL} секунд...")
                await asyncio.sleep(self.CHECK_INTERVAL)

            except asyncio.CancelledError:
                logger.info("🛑 Основной цикл отменен")
                break
            except Exception as e:
                logger.error(f"💥 Критическая ошибка в основном цикле: {e}")
                await self.send_telegram_message(f"💥 Критическая ошибка: {e}")
                await asyncio.sleep(300)  # Пауза 5 минут при ошибке

    async def run(self):
        """Запуск бота"""
        try:
            # Начальная проверка
            available = self.get_available_balance(self.STABLE)
            if available:
                message = (
                    f"🤖 Bybit Funding Bot запущен!\n"
                    f"💰 Баланс: {available:.2f} {self.STABLE}\n"
                    f"📈 Пары: {', '.join(self.SYMBOLS)}\n"
                    f"💼 Размер: {self.POSITION_SIZE} USDT\n"
                    f"📊 Порог: {self.FUNDING_RATE_THRESHOLD}%\n"
                    f"🔄 Интервал: {self.CHECK_INTERVAL/60} мин"
                )
                await self.send_telegram_message(message, parse_mode="HTML")
                logger.info(f"✅ Начальный баланс: {available:.2f} {self.STABLE}")
            else:
                logger.error("❌ Не удалось получить начальный баланс")

            # Запуск основного цикла
            await self.main_loop()

        except KeyboardInterrupt:
            logger.info("👋 Получен сигнал прерывания")
        except Exception as e:
            logger.error(f"💥 Фатальная ошибка: {e}")
            await self.send_telegram_message(f"💥 Фатальная ошибка бота: {e}")
        finally:
            await self._cleanup()

async def main():
    """Точка входа"""
    bot = BybitFundingBot()
    await bot.run()

if __name__ == "__main__":
    # Проверка переменных окружения
    required_vars = ["BYBIT_API_KEY", "BYBIT_API_SECRET", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        print("📋 Установите их с помощью:")
        for var in missing_vars:
            print(f"export {var}='ваш_значение'")
        sys.exit(1)

    # Запуск
    asyncio.run(main())