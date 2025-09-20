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
from dotenv import load_dotenv

# Загрузка переменных из .env
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/bybit_funding_bot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BybitFundingBot:
    def __init__(self):
        # Конфигурация
        self.SYMBOLS = ["ETHUSDT", "DOGEUSDT"]  # Поддерживаемые пары
        self.STABLE = "USDT"
        self.POSITION_SIZE = 5.0  # USDT на каждую позицию
        self.CHECK_INTERVAL = 1800  # 30 минут
        self.FUNDING_RATE_THRESHOLD = 0.02  # 0.02% минимальный funding rate
        self.MAX_POSITIONS_PER_SYMBOL = 1  # Максимум позиций на пару
        self.ORDER_TYPE = "Market"  # "Market" или "Limit"
        self.STOP_LOSS_PERCENT = 0.05  # 5% стоп-лосс
        self.CLOSE_NEGATIVE_RATE = True  # Закрывать при отрицательном funding rate

        # Telegram настройки
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        self.CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

        # API ключи
        self.API_KEY = os.getenv("BYBIT_API_KEY")
        self.API_SECRET = os.getenv("BYBIT_API_SECRET")

        # Проверка обязательных переменных
        required_vars = {
            "BYBIT_API_KEY": self.API_KEY,
            "BYBIT_API_SECRET": self.API_SECRET,
            "TELEGRAM_TOKEN": self.TELEGRAM_TOKEN,
            "TELEGRAM_CHAT_ID": self.CHAT_ID
        }

        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")

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
        self.balance_cache = {}  # Кэш баланса

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Создание папки для логов
        os.makedirs("logs", exist_ok=True)

        logger.info(f"🚀 Bybit Funding Rate Bot v2.0 запущен")
        logger.info(f"💰 Размер позиции: {self.POSITION_SIZE} USDT")
        logger.info(f"📊 Порог funding rate: {self.FUNDING_RATE_THRESHOLD}%")
        logger.info(f"🔄 Интервал проверки: {self.CHECK_INTERVAL//60} мин")

    def _signal_handler(self, signum, frame):
        """Обработка сигналов для graceful shutdown"""
        logger.info(f"Получен сигнал {signum}. Завершение работы...")
        self.running = False
        asyncio.create_task(self._cleanup())

    async def _cleanup(self):
        """Очистка при завершении работы"""
        logger.info("🔄 Закрытие всех активных позиций...")
        close_tasks = [self._close_position(symbol) for symbol in list(self.active_positions.keys())]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        logger.info("✅ Бот остановлен корректно")
        sys.exit(0)

    async def send_telegram_message(self, message: str, parse_mode: str = None):
        """Отправка уведомления в Telegram"""
        try:
            await self.bot.send_message(
                chat_id=self.CHAT_ID,
                text=message,
                parse_mode=parse_mode
            )
            logger.info(f"📱 Telegram отправлено: {message[:100]}...")
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
            qty_step = lot_size_filter.get("qtyStep", "0.0001")
            qty_precision = len(qty_step.split(".")[-1]) if "." in qty_step else 0
            min_order_amt = float(lot_size_filter.get("minOrderAmt", "10")) if category == "spot" else 0.0

            logger.debug(f"📊 {category.upper()} {symbol}: minQty={min_order_qty}, precision={qty_precision}, minAmt={min_order_amt}")
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
            logger.debug(f"💹 {symbol}: Funding Rate {rate_percent:.4f}%")
            return rate_percent
        except Exception as e:
            logger.error(f"❌ Ошибка получения funding rate для {symbol}: {e}")
            return None

    def get_spot_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены на споте"""
        try:
            ticker = self.session.get_tickers(category="spot", symbol=symbol)
            if ticker.get("retCode") != 0:
                logger.error(f"❌ Ошибка получения спотовой цены {symbol}: {ticker.get('retMsg')}")
                return None
            return float(ticker["result"]["list"][0]["lastPrice"])
        except Exception as e:
            logger.error(f"❌ Ошибка получения спотовой цены {symbol}: {e}")
            return None

    def get_perp_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены на перпетуале"""
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            if ticker.get("retCode") != 0:
                logger.error(f"❌ Ошибка получения цены перпетуала {symbol}: {ticker.get('retMsg')}")
                return None
            return float(ticker["result"]["list"][0]["lastPrice"])
        except Exception as e:
            logger.error(f"❌ Ошибка получения цены перпетуала {symbol}: {e}")
            return None

    def get_available_balance(self, coin: str) -> Optional[float]:
        """Проверка доступного баланса с поддержкой разных типов аккаунтов"""
        # Кэш на 5 минут
        cache_key = f"{coin}_{int(time.time() // 300)}"
        if cache_key in self.balance_cache:
            return self.balance_cache[cache_key]

        account_types = ["UNIFIED", "FUND", "SPOT"]  # Пробуем разные типы

        for account_type in account_types:
            try:
                logger.debug(f"🔍 Пробуем accountType={account_type} для {coin}")

                # Сначала попробуем без параметра coin для получения всех монет
                if coin == "USDT":
                    balance = self.session.get_wallet_balance(accountType=account_type)
                else:
                    balance = self.session.get_wallet_balance(accountType=account_type, coin=coin)

                logger.debug(f"🔍 {account_type} ответ API: {json.dumps(balance, indent=2)[:500]}...")

                if balance.get("retCode") != 0:
                    logger.debug(f"ℹ️  {account_type}: API ошибка - {balance.get('retMsg')}")
                    continue

                # Проверяем структуру ответа
                result_list = balance["result"]["list"]
                if not result_list:
                    logger.debug(f"ℹ️  {account_type}: Список пуст")
                    continue

                # Для разных типов аккаунта структура может отличаться
                if account_type == "SPOT":
                    # Spot аккаунт имеет другую структуру
                    for account in result_list:
                        for c in account.get("coin", []):
                            if c["coin"] == coin:
                                balance_value = c.get("free", c.get("walletBalance", "0"))
                                if balance_value:
                                    balance_amount = float(balance_value)
                                    logger.info(f"💰 {account_type} баланс {coin}: {balance_amount:.2f}")
                                    self.balance_cache[cache_key] = balance_amount
                                    return balance_amount
                else:
                    # Unified/Fund аккаунты
                    for account in result_list:
                        coin_list = account.get("coin", [])
                        for c in coin_list:
                            if c["coin"] == coin:
                                # Пробуем разные поля в зависимости от типа аккаунта
                                balance_value = (
                                    c.get("walletBalance") or
                                    c.get("availableToWithdraw") or
                                    c.get("free") or
                                    c.get("equity") or
                                    c.get("totalEquity") or
                                    "0"
                                )
                                if balance_value:
                                    balance_amount = float(balance_value)
                                    logger.info(f"💰 {account_type} баланс {coin}: {balance_amount:.2f}")
                                    self.balance_cache[cache_key] = balance_amount
                                    return balance_amount

                logger.debug(f"ℹ️  {account_type}: {coin} не найден в списке монет")

            except Exception as e:
                logger.debug(f"ℹ️  {account_type}: Исключение - {e}")
                continue

        logger.error(f"❌ Не удалось получить баланс {coin} ни для одного типа аккаунта")
        self.balance_cache[cache_key] = 0.0
        return 0.0

    def calculate_qty(self, position_size: float, price: float, min_order_qty: float, qty_precision: int) -> float:
        """Расчет количества с учетом ограничений"""
        if price <= 0:
            logger.error(f"❌ Неверная цена: {price}")
            return 0.0

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
                logger.warning(f"⚠️  {symbol} Spot: value {order_value:.2f} < min {min_order_amt:.2f}")
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

            logger.info(f"📈 {symbol} Spot {side}: qty={qty:.6f}, value={order_value:.2f} USDT")
            response = self.session.place_order(**order_params)

            if response.get("retCode") != 0:
                error_msg = f"❌ {symbol} Spot {self.ORDER_TYPE} ошибка: {response.get('retMsg')} (Code: {response.get('retCode')})"
                logger.error(error_msg)
                await self.send_telegram_message(error_msg)
                return None

            order_id = response["result"]["orderId"]
            logger.info(f"✅ {symbol} Spot ордер размещен: {order_id}")
            await self.send_telegram_message(f"✅ {symbol} Spot: {qty:.6f} по {spot_price:,.2f} USDT")
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

            logger.info(f"📉 {symbol} Perp {side}: qty={qty:.4f}")
            response = self.session.place_order(**order_params)

            if response.get("retCode") != 0:
                error_msg = f"❌ {symbol} Perp {self.ORDER_TYPE} ошибка: {response.get('retMsg')} (Code: {response.get('retCode')})"
                logger.error(error_msg)
                await self.send_telegram_message(error_msg)
                return None

            order_id = response["result"]["orderId"]
            logger.info(f"✅ {symbol} Perp ордер размещен: {order_id}")
            await self.send_telegram_message(f"✅ {symbol} Perp: {qty:.4f} по {perp_price:,.2f} USDT")

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
            if side == "Sell":  # Short позиция (мы продаем перпетуал)
                stop_price = entry_price * (1 + self.STOP_LOSS_PERCENT)  # Закрытие при росте цены
                stop_side = "Buy"  # Покупаем для закрытия short
            else:  # Long позиция
                stop_price = entry_price * (1 - self.STOP_LOSS_PERCENT)
                stop_side = "Sell"

            # Для Bybit V5 используем conditional order
            stop_params = {
                "category": "linear",
                "symbol": symbol,
                "side": stop_side,
                "orderType": "Market",
                "qty": str(qty),
                "triggerPrice": str(stop_price),
                "triggerBy": "LastPrice",
                "orderLinkId": f"{symbol}_stoploss_{int(time.time())}",
                "triggerDirection": 1 if side == "Sell" else 0  # 1 для short, 0 для long
            }

            response = self.session.place_order(**stop_params)

            if response.get("retCode") == 0:
                logger.info(f"🛡️ {symbol} Stop-loss установлен: {stop_price:.2f}")
                await self.send_telegram_message(f"🛡️ {symbol} Stop-loss: {stop_price:.2f}")
            else:
                logger.warning(f"⚠️  {symbol} Не удалось установить stop-loss: {response.get('retMsg')}")

        except Exception as e:
            logger.error(f"❌ {symbol} Ошибка stop-loss: {e}")

    async def _close_position(self, symbol: str) -> bool:
        """Закрытие арбитражной позиции"""
        try:
            if symbol not in self.active_positions:
                logger.info(f"ℹ️  {symbol}: Позиция не найдена для закрытия")
                return True

            position = self.active_positions[symbol]
            close_time = datetime.now()
            duration = (close_time - position["open_time"]).total_seconds() / 3600  # в часах

            success = True

            # Закрытие спотовой позиции
            if "spot_order_id" in position:
                try:
                    # Для спота используем market sell
                    spot_close = self.session.place_order(
                        category="spot",
                        symbol=symbol,
                        side="Sell",
                        orderType="Market",
                        qty=str(position["qty"])
                    )
                    if spot_close.get("retCode") == 0:
                        logger.info(f"✅ {symbol} Spot закрыт")
                    else:
                        logger.error(f"❌ {symbol} Spot закрытие ошибка: {spot_close.get('retMsg')}")
                        success = False
                except Exception as e:
                    logger.error(f"❌ {symbol} Spot закрытие исключение: {e}")
                    success = False

            # Закрытие перпетуальной позиции
            if "perp_order_id" in position:
                try:
                    perp_close = self.session.close_position(
                        category="linear",
                        symbol=symbol
                    )
                    if perp_close.get("retCode") == 0:
                        logger.info(f"✅ {symbol} Perp закрыт")
                    else:
                        logger.error(f"❌ {symbol} Perp закрытие ошибка: {perp_close.get('retMsg')}")
                        success = False
                except Exception as e:
                    logger.error(f"❌ {symbol} Perp закрытие исключение: {e}")
                    success = False

            # Удаление из активных позиций
            del self.active_positions[symbol]

            reason = "по расписанию" if duration > 24 else "funding rate"
            message = (
                f"🔒 {symbol} позиция закрыта\n"
                f"⏱️  Время работы: {duration:.1f}ч\n"
                f"📊 Причина: {reason}"
            )
            await self.send_telegram_message(message)
            logger.info(f"✅ {symbol}: Позиция закрыта ({duration:.1f}ч)")

            return success

        except Exception as e:
            logger.error(f"❌ {symbol} Критическая ошибка закрытия: {e}")
            return False

    async def check_existing_positions(self, symbol: str) -> bool:
        """Проверка существующих позиций"""
        try:
            # Проверка перпетуальной позиции
            positions = self.session.get_positions(category="linear", symbol=symbol)
            if positions.get("retCode") != 0:
                logger.error(f"❌ {symbol} Ошибка проверки позиций: {positions.get('retMsg')}")
                return False

            for pos in positions["result"]["list"]:
                size = float(pos["size"])
                if size > 0:
                    logger.info(f"ℹ️  {symbol}: Существующая позиция {size}, пропускаем")
                    return True

            # Проверка открытых ордеров на споте
            spot_orders = self.session.get_order_history(category="spot", symbol=symbol, limit=10)
            if spot_orders.get("retCode") == 0:
                for order in spot_orders["result"]["list"]:
                    if order["orderStatus"] in ["New", "PartiallyFilled"]:
                        logger.info(f"ℹ️  {symbol}: Открытый спотовый ордер {order['orderId']}, пропускаем")
                        return True

            return False
        except Exception as e:
            logger.error(f"❌ {symbol} Ошибка проверки позиций: {e}")
            return False

    async def open_arbitrage_position(self, symbol: str) -> bool:
        """Открытие арбитражной позиции"""
        try:
            logger.info(f"🎯 {symbol}: Попытка открытия арбитражной позиции")

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

            # Проверка баланса
            available = self.get_available_balance(self.STABLE)
            if available is None or available < self.POSITION_SIZE * 2:
                balance_str = f"{available:.2f}" if available is not None else "N/A"
                logger.warning(f"⚠️  {symbol}: Недостаточно баланса ({balance_str} < {self.POSITION_SIZE * 2:.2f})")
                await self.send_telegram_message(f"⚠️  Низкий баланс для {symbol}: {balance_str} USDT")
                return False

            # Расчет количества
            qty = self.calculate_qty(
                self.POSITION_SIZE, spot_price,
                spot_info["minOrderQty"], spot_info["qtyPrecision"]
            )

            order_value = qty * spot_price
            if order_value < spot_info["minOrderAmt"]:
                logger.warning(f"⚠️  {symbol}: Стоимость {order_value:.2f} < min {spot_info['minOrderAmt']:.2f}")
                return False

            logger.info(f"📊 {symbol}: qty={qty:.6f}, value={order_value:.2f} USDT")

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
                try:
                    self.session.cancel_order(category="spot", symbol=symbol, orderId=spot_order_id)
                    logger.info(f"🔄 {symbol}: Спотовый ордер отменен")
                except Exception as e:
                    logger.error(f"❌ {symbol}: Ошибка отмены спотового ордера: {e}")
                return False

            # Сохранение активной позиции
            self.active_positions[symbol] = {
                "spot_order_id": spot_order_id,
                "perp_order_id": perp_order_id,
                "open_time": datetime.now(),
                "spot_price": spot_price,
                "perp_price": perp_price,
                "qty": qty,
                "funding_rate_at_open": self.get_funding_rate(symbol)
            }

            message = (
                f"🚀 <b>{symbol}</b> Арбитражная позиция открыта!\n"
                f"📈 <b>Спот BUY</b>: {qty:.6f} по {spot_price:,.2f}\n"
                f"📉 <b>Perp SELL</b>: {qty:.6f} по {perp_price:,.2f}\n"
                f"💰 <b>Размер</b>: {self.POSITION_SIZE} USDT\n"
                f"🔒 <b>Stop-loss</b>: {self.STOP_LOSS_PERCENT*100}%\n"
                f"⏰ <b>{datetime.now().strftime('%H:%M:%S')}</b>"
            )
            await self.send_telegram_message(message, parse_mode="HTML")
            logger.info(f"✅ {symbol}: Арбитражная позиция открыта успешно")

            return True

        except Exception as e:
            logger.error(f"❌ {symbol}: Критическая ошибка открытия позиции: {e}")
            await self.send_telegram_message(f"💥 {symbol}: Ошибка открытия позиции: {e}")
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

                # Периодический статус (каждые 8 часов)
                if int(duration) % 8 == 0 and duration > 0.1:
                    current_fr = self.get_funding_rate(symbol)
                    if current_fr:
                        profit_estimate = current_fr * self.POSITION_SIZE * duration / 100
                        message = (
                            f"📊 <b>{symbol}</b> статус\n"
                            f"⏱️  <b>Время</b>: {duration:.1f}ч\n"
                            f"💹 <b>Funding</b>: {current_fr:.4f}%\n"
                            f"💵 <b>Оценка прибыли</b>: ~{profit_estimate:.2f} USDT"
                        )
                        await self.send_telegram_message(message, parse_mode="HTML")

        except Exception as e:
            logger.error(f"❌ Ошибка мониторинга позиций: {e}")

    async def check_balance_alert(self):
        """Проверка баланса и алерты"""
        try:
            available = self.get_available_balance(self.STABLE)
            if available is None:
                return

            total_required = len(self.SYMBOLS) * self.POSITION_SIZE * 2
            if available < total_required * 0.3:  # Менее 30% от требуемого
                message = (
                    f"⚠️ <b>КРИТИЧЕСКИ НИЗКИЙ БАЛАНС!</b>\n"
                    f"💰 <b>Доступно</b>: {available:.2f} {self.STABLE}\n"
                    f"📊 <b>Требуется</b>: {total_required:.2f} {self.STABLE}\n"
                    f"🔴 <b>ОПАСНО!</b> Рекомендуется пополнение"
                )
                await self.send_telegram_message(message, parse_mode="HTML")
                logger.critical(f"🚨 КРИТИЧЕСКИ НИЗКИЙ БАЛАНС: {available:.2f} USDT")
            elif available < total_required * 0.5:  # Менее 50%
                message = (
                    f"⚠️ <b>Низкий баланс</b>\n"
                    f"💰 <b>Доступно</b>: {available:.2f} {self.STABLE}\n"
                    f"📊 <b>Требуется</b>: {total_required:.2f} {self.STABLE}\n"
                    f"🟡 Рекомендуется пополнение"
                )
                await self.send_telegram_message(message, parse_mode="HTML")
                logger.warning(f"⚠️ Низкий баланс: {available:.2f} USDT")

        except Exception as e:
            logger.error(f"❌ Ошибка проверки баланса: {e}")

    async def main_loop(self):
        """Основной цикл бота"""
        logger.info("🔄 Запуск основного цикла...")
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.running:
            try:
                # Сброс счетчика ошибок
                consecutive_errors = 0

                # Проверка баланса
                await self.check_balance_alert()

                # Мониторинг позиций
                await self.monitor_positions()

                # Проверка новых возможностей
                for symbol in self.SYMBOLS:
                    if symbol in self.active_positions and len(self.active_positions) >= self.MAX_POSITIONS_PER_SYMBOL * len(self.SYMBOLS):
                        continue

                    funding_rate = self.get_funding_rate(symbol)
                    if funding_rate is None:
                        continue

                    logger.info(f"📊 {symbol}: Funding Rate {funding_rate:.4f}%")

                    if funding_rate > self.FUNDING_RATE_THRESHOLD:
                        logger.info(f"🎯 {symbol}: Funding rate {funding_rate:.4f}% > порога {self.FUNDING_RATE_THRESHOLD}%")
                        success = await self.open_arbitrage_position(symbol)
                        if success:
                            await asyncio.sleep(60)  # Пауза после открытия позиции
                    elif funding_rate < 0 and symbol in self.active_positions:
                        logger.info(f"📉 {symbol}: Funding rate отрицательный ({funding_rate:.2f}%), закрываем")
                        await self._close_position(symbol)

                # Пауза между циклами
                logger.debug(f"😴 Пауза {self.CHECK_INTERVAL} секунд...")
                await asyncio.sleep(self.CHECK_INTERVAL)

            except asyncio.CancelledError:
                logger.info("🛑 Основной цикл отменен")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"💥 Ошибка в основном цикле #{consecutive_errors}: {e}")

                # Критическая ошибка - уведомление
                if consecutive_errors >= 3:
                    await self.send_telegram_message(f"💥 КРИТИЧЕСКАЯ ОШИБКА #{consecutive_errors}: {e}")

                # Слишком много ошибок подряд - пауза
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"🚨 КРИТИЧЕСКИЙ СБОЙ: {consecutive_errors} ошибок подряд. Пауза 30 мин.")
                    await self.send_telegram_message(f"🚨 КРИТИЧЕСКИЙ СБОЙ: {consecutive_errors} ошибок. Пауза 30 мин.")
                    await asyncio.sleep(1800)  # 30 минут
                    consecutive_errors = 0
                else:
                    await asyncio.sleep(300)  # 5 минут при ошибке

    async def run(self):
        """Запуск бота"""
        try:
            # Начальная проверка
            logger.info("🔍 Проверка начального баланса...")
            available = self.get_available_balance(self.STABLE)

            # Добавляем отладочную информацию
            if available is None:
                logger.error("❌ Не удалось получить баланс - возвращен None")
                # Попробуем получить полную информацию о балансе
                try:
                    logger.info("🔍 Отладка: запрос баланса UNIFIED без coin...")
                    debug_balance = self.session.get_wallet_balance(accountType="UNIFIED")
                    logger.info(f"🔍 Полная информация о балансе UNIFIED: {json.dumps(debug_balance, indent=2)[:1000]}...")

                    logger.info("🔍 Отладка: запрос баланса SPOT без coin...")
                    debug_balance_spot = self.session.get_wallet_balance(accountType="SPOT")
                    logger.info(f"🔍 Полная информация о балансе SPOT: {json.dumps(debug_balance_spot, indent=2)[:1000]}...")

                except Exception as debug_e:
                    logger.error(f"❌ Ошибка отладки баланса: {debug_e}")
                    logger.error(f"🔍 Ответ API: {debug_e}")
            else:
                logger.info(f"💰 Начальный баланс: {available:.2f} {self.STABLE}")

            # ИСПРАВЛЕНО: правильное форматирование строки
            balance_display = f"{available:.2f}" if available is not None else "N/A"
            message_parts = [
                f"🤖 <b>Bybit Funding Bot v2.0</b> запущен!",
                f"💰 <b>Баланс</b>: {balance_display} {self.STABLE}",
                f"📈 <b>Пары</b>: {', '.join(self.SYMBOLS)}",
                f"💼 <b>Размер</b>: {self.POSITION_SIZE} USDT",
                f"📊 <b>Порог</b>: {self.FUNDING_RATE_THRESHOLD}%",
                f"🔄 <b>Интервал</b>: {self.CHECK_INTERVAL//60} мин",
                f"🛡️ <b>Stop-loss</b>: {self.STOP_LOSS_PERCENT*100}%"
            ]

            startup_message = "\n".join(message_parts)
            await self.send_telegram_message(startup_message, parse_mode="HTML")

            if available is not None and available > 0:
                logger.info(f"✅ Начальный баланс: {available:.2f} {self.STABLE}")
            elif available is None:
                logger.warning("⚠️  Не удалось получить начальный баланс")
                await self.send_telegram_message("⚠️  Не удалось получить баланс. Проверьте API ключи и тип аккаунта.")
            else:
                logger.warning(f"⚠️  Начальный баланс равен 0: {available:.2f} {self.STABLE}")

            # Запуск основного цикла
            await self.main_loop()

        except KeyboardInterrupt:
            logger.info("👋 Получен сигнал прерывания от пользователя")
        except Exception as e:
            logger.critical(f"💥 Фатальная ошибка бота: {e}")
            await self.send_telegram_message(f"💥 <b>ФАТАЛЬНАЯ ОШИБКА БОТА</b>\n{e}", parse_mode="HTML")
        finally:
            await self._cleanup()

async def main():
    """Точка входа"""
    try:
        bot = BybitFundingBot()
        await bot.run()
    except ValueError as e:
        print(f"❌ Конфигурация: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())