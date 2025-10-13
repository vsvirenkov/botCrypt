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
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f'logs/bybit_bot_{datetime.now().strftime("%Y%m%d_%H%M")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('BybitBot')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("🚀 === ЛОГИРОВАНИЕ ЗАГРУЖЕНО ===")

class BybitFundingBot:
    def __init__(self):
        logger.info("🔧 === ИНИЦИАЛИЗАЦИЯ BybitFundingBot ===")

        # Конфигурация - Funding Arbitrage
        self.SYMBOLS = ["ETHUSDT", "DOGEUSDT", "SOLUSDT", "BNBUSDT"]
        self.STABLE = "USDT"
        self.POSITION_SIZE = 5.0
        self.CHECK_INTERVAL = 1800
        self.FUNDING_RATE_THRESHOLD = 0.02
        self.MAX_POSITIONS_PER_SYMBOL = 1
        self.ORDER_TYPE = "Market"
        self.STOP_LOSS_PERCENT = 0.05
        self.CLOSE_NEGATIVE_RATE = True

        # Конфигурация - Scalping с STOP LOSS
        self.SCALP_SYMBOLS = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]
        self.SCALP_POSITION_SIZE = 5.0
        self.SCALP_CHECK_INTERVAL = 30
        self.SCALP_PROFIT_TARGET = 0.01  # 0.3%
        self.SCALP_STOP_LOSS = 0.01      # 1%
        self.SCALP_TRAILING_STOP = 0.001 # 0.1%
        self.SCALP_RSI_PERIOD = 14
        self.SCALP_RSI_OVERSOLD = 30
        self.SCALP_RSI_OVERBOUGHT = 70
        self.SCALP_VOLUME_MULTIPLIER = 1.5
        self.SCALP_MAX_POSITIONS = 3
        self.SCALP_TIMEOUT_MINUTES = 300
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        self.MACD_THRESHOLD = 0.0  # Минимальная разница между MACD и Signal Line для сигнала

        # Мониторинг
        self.SCALP_STATUS_INTERVAL = 300
        self.TELEGRAM_STATUS_INTERVAL = 1800

        # Режим работы
        self.BOT_MODE = "scalping"

        logger.info(f"⚙️  РЕЖИМ: {self.BOT_MODE.upper()}")
        logger.info(f"📈 СКАЛЬПИНГ ПАРЫ: {', '.join(self.SCALP_SYMBOLS)}")
        logger.info(f"🔄 ИНТЕРВАЛ: {self.SCALP_CHECK_INTERVAL} сек")
        logger.info(f"🛡️ STOP LOSS: {self.SCALP_STOP_LOSS*100:.1f}% | Тейк: {self.SCALP_PROFIT_TARGET*100:.1f}%")
        logger.info(f"📊 MACD: Fast={self.MACD_FAST}, Slow={self.MACD_SLOW}, Signal={self.MACD_SIGNAL}")

        # Telegram настройки
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        self.CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

        # API ключи
        self.API_KEY = os.getenv("BYBIT_API_KEY")
        self.API_SECRET = os.getenv("BYBIT_API_SECRET")

        # Проверка переменных
        required_vars = {
            "BYBIT_API_KEY": self.API_KEY,
            "BYBIT_API_SECRET": self.API_SECRET,
            "TELEGRAM_TOKEN": self.TELEGRAM_TOKEN,
            "TELEGRAM_CHAT_ID": self.CHAT_ID
        }

        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            logger.error(f"❌ ОТСУТСТВУЮТ: {', '.join(missing_vars)}")
            raise ValueError(f"Отсутствуют переменные: {', '.join(missing_vars)}")

        logger.info("✅ ВСЕ ПЕРЕМЕННЫЕ НА МЕСТЕ")

        # Инициализация клиентов
        try:
            self.session = HTTP(
                api_key=self.API_KEY,
                api_secret=self.API_SECRET,
                testnet=False
            )
            logger.info("🔗 Bybit API подключен")
        except Exception as e:
            logger.error(f"❌ API ОШИБКА: {e}")
            raise

        try:
            self.bot = telegram.Bot(token=self.TELEGRAM_TOKEN)
            logger.info("📱 Telegram бот готов")
        except Exception as e:
            logger.error(f"❌ TELEGRAM ОШИБКА: {e}")
            raise

        # Состояние бота
        self.active_positions = {}
        self.active_scalp_positions = {}
        self.ohlcv_cache = {}
        self.rsi_cache = {}
        self.macd_cache = {}
        self.symbol_info_cache = {}
        self.balance_cache = {}
        self.stop_loss_orders = {}
        self.take_profit_orders = {}
        self.running = True
        self.last_scalp_check = 0
        self.last_status_update = 0
        self.last_telegram_status = 0
        self.signal_checks = 0
        self.successful_signals = 0

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("🛡️ Signal handlers установлены")

        # Папка для логов
        os.makedirs("logs", exist_ok=True)
        logger.info("📁 Папка logs готова")

        # Тест API
        self._test_api_connection()
        self._validate_symbols()
        logger.info("✅ === ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА ===")

    def _test_api_connection(self):
        """Тест подключения API"""
        logger.info("🔍 ТЕСТ API...")
        try:
            test_response = self.session.get_tickers(category="linear", symbol="BTCUSDT")
            if test_response.get("retCode") == 0:
                price = float(test_response["result"]["list"][0]["lastPrice"])
                logger.info(f"✅ API OK | BTCUSDT: ${price:,.2f}")
            else:
                logger.warning(f"⚠️  API ТЕСТ: {test_response.get('retMsg')}")
        except Exception as e:
            logger.error(f"❌ API ТЕСТ: {e}")

    def _validate_symbols(self):
        """Проверка символов"""
        logger.info("🔍 ПРОВЕРКА СИМВОЛОВ...")
        all_symbols = list(set(self.SYMBOLS + self.SCALP_SYMBOLS))
        valid_symbols = []

        for symbol in all_symbols:
            try:
                response = self.session.get_instruments_info(category="linear", symbol=symbol)
                if response.get("retCode") == 0 and response["result"]["list"]:
                    logger.info(f"✅ {symbol} - OK")
                    valid_symbols.append(symbol)
                    self.symbol_info_cache[symbol] = {
                        "linear": True,
                        "spot": True,
                        "last_check": time.time()
                    }
                else:
                    logger.warning(f"⚠️  {symbol} - ОТКЛОНЕН")
            except Exception as e:
                logger.warning(f"⚠️  {symbol} - ОШИБКА: {e}")

        self.SYMBOLS = [s for s in self.SYMBOLS if s in valid_symbols]
        self.SCALP_SYMBOLS = [s for s in self.SCALP_SYMBOLS if s in valid_symbols]

        logger.info(f"📊 ВАЛИДНЫЕ: {valid_symbols}")
        logger.info(f"📈 СКАЛЬПИНГ: {self.SCALP_SYMBOLS}")

        if self.BOT_MODE == "scalping" and not self.SCALP_SYMBOLS:
            raise ValueError("Нет символов для скальпинга!")

    def is_symbol_valid(self, symbol: str, category: str = "linear") -> bool:
        if symbol in self.symbol_info_cache:
            info = self.symbol_info_cache[symbol]
            if time.time() - info["last_check"] < 3600:
                return info.get(category, False)

        try:
            response = self.session.get_instruments_info(category=category, symbol=symbol)
            is_valid = response.get("retCode") == 0 and response["result"]["list"]
            self.symbol_info_cache[symbol] = {
                category: is_valid,
                "last_check": time.time()
            }
            return is_valid
        except Exception:
            return False

    def get_available_balance(self, coin: str, account_type: str = "UNIFIED") -> Optional[float]:
        cache_key = f"{coin}_{int(time.time() // 300)}"
        if cache_key in self.balance_cache:
            return self.balance_cache[cache_key]

        account_types = ["UNIFIED", "FUND", "SPOT"]

        for account_type in account_types:
            try:
                if coin == "USDT":
                    balance = self.session.get_wallet_balance(accountType=account_type)
                else:
                    balance = self.session.get_wallet_balance(accountType=account_type, coin=coin)

                if balance.get("retCode") != 0:
                    continue

                result_list = balance["result"]["list"]
                if not result_list:
                    continue

                if account_type == "SPOT":
                    for account in result_list:
                        for c in account.get("coin", []):
                            if c["coin"] == coin:
                                balance_value = c.get("free", c.get("walletBalance", "0"))
                                if balance_value:
                                    balance_amount = float(balance_value)
                                    self.balance_cache[cache_key] = balance_amount
                                    return balance_amount
                else:
                    for account in result_list:
                        coin_list = account.get("coin", [])
                        for c in coin_list:
                            if c["coin"] == coin:
                                balance_value = (
                                    c.get("walletBalance") or
                                    c.get("availableToWithdraw") or
                                    c.get("free") or
                                    c.get("equity") or
                                    "0"
                                )
                                if balance_value and balance_value != "0":
                                    balance_amount = float(balance_value)
                                    self.balance_cache[cache_key] = balance_amount
                                    return balance_amount

            except Exception:
                continue

        self.balance_cache[cache_key] = 0.0
        return 0.0

    def get_instrument_info(self, category: str, symbol: str) -> Optional[Dict]:
        if not self.is_symbol_valid(symbol, category):
            return None
        try:
            response = self.session.get_instruments_info(category=category, symbol=symbol)
            if response.get("retCode") != 0:
                return None

            instrument = response["result"]["list"][0]
            lot_size_filter = instrument.get("lotSizeFilter", {})
            price_filter = instrument.get("priceFilter", {})
            min_order_qty = float(lot_size_filter.get("minOrderQty", "0.001"))
            qty_step = lot_size_filter.get("qtyStep", "0.0001")
            qty_precision = len(qty_step.split(".")[-1]) if "." in qty_step else 0
            price_step = price_filter.get("tickSize", "0.01")
            price_precision = len(price_step.split(".")[-1]) if "." in price_step else 0

            return {
                "minOrderQty": min_order_qty,
                "qtyPrecision": qty_precision,
                "pricePrecision": price_precision,
                "minOrderAmt": 10.0 if category == "spot" else 0.0
            }
        except Exception:
            return None

    def get_ohlcv(self, symbol: str, interval: str = "1", limit: int = 100) -> Optional[List]:
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            if response.get("retCode") != 0:
                return None

            klines = response["result"]["list"]
            ohlcv_data = []
            for kline in klines:
                ohlcv_data.append({
                    "timestamp": int(kline[0]),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5])
                })
            return ohlcv_data
        except Exception:
            return None

    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Рассчитать EMA (экспоненциальное скользящее среднее)"""
        if len(prices) < period:
            return []
        ema = []
        multiplier = 2 / (period + 1)
        ema.append(sum(prices[:period]) / period)  # Начальная SMA
        for price in prices[period:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
        return ema

    def calculate_macd(self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict]:
        """Рассчитать MACD и Signal Line"""
        try:
            ohlcv = self.get_ohlcv(symbol, "1", max(fast, slow, signal) + 10)
            if not ohlcv or len(ohlcv) < max(fast, slow, signal) + 1:
                return None

            closes = [candle["close"] for candle in ohlcv[::-1]]  # Перевернуть для правильного порядка
            ema_fast = self.calculate_ema(closes, fast)
            ema_slow = self.calculate_ema(closes, slow)

            if len(ema_fast) < 1 or len(ema_slow) < 1:
                return None

            macd = [f - s for f, s in zip(ema_fast[-len(ema_slow):], ema_slow)]
            signal_line = self.calculate_ema(macd, signal)

            if len(signal_line) < 1:
                return None

            return {
                "macd": macd[-1],
                "signal": signal_line[-1],
                "histogram": macd[-1] - signal_line[-1]
            }
        except Exception as e:
            logger.warning(f"⚠️ Ошибка расчета MACD для {symbol}: {e}")
            return None

    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_rsi(self, symbol: str) -> Optional[float]:
        try:
            ohlcv = self.get_ohlcv(symbol, "1", 50)
            if not ohlcv or len(ohlcv) < 20:
                return None

            closes = [candle["close"] for candle in ohlcv[-20:]]
            return self.calculate_rsi(closes, self.SCALP_RSI_PERIOD)
        except Exception:
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            if ticker.get("retCode") != 0:
                return None
            return float(ticker["result"]["list"][0]["lastPrice"])
        except Exception:
            return None

    def get_volume_info(self, symbol: str) -> Optional[Dict]:
        try:
            ohlcv = self.get_ohlcv(symbol, "1", 20)
            if not ohlcv:
                return None

            current_volume = ohlcv[0]["volume"]
            avg_volume = sum(candle["volume"] for candle in ohlcv[1:]) / len(ohlcv[1:])

            return {
                "current": current_volume,
                "average": avg_volume,
                "multiplier": current_volume / avg_volume if avg_volume > 0 else 1.0
            }
        except Exception:
            return {"multiplier": 1.0}

    def calculate_scalp_qty(self, symbol: str, position_size: float) -> Optional[float]:
        try:
            instrument_info = self.get_instrument_info("linear", symbol)
            if not instrument_info:
                return None

            price = self.get_current_price(symbol)
            if not price or price <= 0:
                return None

            qty = position_size / price
            return max(round(qty, instrument_info["qtyPrecision"]), instrument_info["minOrderQty"])
        except Exception:
            return None

    async def _set_scalp_risk_management(self, symbol: str, side: str, qty: float, entry_price: float) -> bool:
        """Установка Stop Loss и Take Profit для скальп позиции"""
        logger.info(f"🛡️ УСТАНОВКА РИСК-МАНАДЖМЕНТА: {symbol} {side}")

        try:
            instrument_info = self.get_instrument_info("linear", symbol)
            if not instrument_info:
                logger.error(f"❌ Ошибка получения информации о {symbol}")
                return False

            price_precision = instrument_info.get("pricePrecision", 4)

            if side == "Buy":
                stop_price = entry_price * (1 - self.SCALP_STOP_LOSS)
                take_profit_price = entry_price * (1 + self.SCALP_PROFIT_TARGET)
                stop_side = "Sell"
                tp_side = "Sell"
                stop_trigger_direction = 2
                tp_trigger_direction = 1
            else:
                stop_price = entry_price * (1 + self.SCALP_STOP_LOSS)
                take_profit_price = entry_price * (1 - self.SCALP_PROFIT_TARGET)
                stop_side = "Buy"
                tp_side = "Buy"
                stop_trigger_direction = 1
                tp_trigger_direction = 2

            stop_price = round(stop_price, price_precision)
            take_profit_price = round(take_profit_price, price_precision)

            logger.info(f"📊 {symbol} | Вход: ${entry_price:,.4f} | SL: ${stop_price:,.4f} | TP: ${take_profit_price:,.4f}")

            stop_params = {
                "category": "linear",
                "symbol": symbol,
                "side": stop_side,
                "orderType": "Market",
                "qty": str(qty),
                "triggerPrice": str(stop_price),
                "triggerBy": "LastPrice",
                "orderLinkId": f"{symbol}_SL_{int(time.time())}",
                "triggerDirection": stop_trigger_direction,
                "timeInForce": "GTC"
            }

            logger.info(f"🛑 РАЗМЕЩАЕМ STOP LOSS: {symbol} {stop_side} | ${stop_price:,.4f}")
            stop_response = self.session.place_order(**stop_params)
            if stop_response.get("retCode") == 0:
                stop_order_id = stop_response["result"]["orderId"]
                self.stop_loss_orders[symbol] = stop_order_id
                logger.info(f"🛑 STOP LOSS #{stop_order_id} | {symbol} {stop_side} | ${stop_price:,.4f}")
            else:
                logger.error(f"❌ STOP LOSS ОШИБКА {symbol}: {stop_response.get('retMsg')} (#{stop_response.get('retCode')})")
                return False

            tp_params = {
                "category": "linear",
                "symbol": symbol,
                "side": tp_side,
                "orderType": "Market",
                "qty": str(qty),
                "triggerPrice": str(take_profit_price),
                "triggerBy": "LastPrice",
                "orderLinkId": f"{symbol}_TP_{int(time.time())}",
                "triggerDirection": tp_trigger_direction,
                "timeInForce": "GTC"
            }

            logger.info(f"🎯 РАЗМЕЩАЕМ TAKE PROFIT: {symbol} {tp_side} | ${take_profit_price:,.4f}")
            tp_response = self.session.place_order(**tp_params)
            if tp_response.get("retCode") == 0:
                tp_order_id = tp_response["result"]["orderId"]
                self.take_profit_orders[symbol] = tp_order_id
                logger.info(f"🎯 TAKE PROFIT #{tp_order_id} | {symbol} {tp_side} | ${take_profit_price:,.4f}")
            else:
                logger.error(f"❌ TAKE PROFIT ОШИБКА {symbol}: {tp_response.get('retMsg')} (#{tp_response.get('retCode')})")
                return False

            risk_msg = (
                f"🛡️ <b>РИСК-МАНАДЖМЕНТ {symbol}</b>\n\n"
                f"📈 <b>Вход</b>: <code>${entry_price:,.4f}</code>\n"
                f"🛑 <b>Stop Loss</b>: <code>${stop_price:,.4f}</code> (-{self.SCALP_STOP_LOSS*100:.1f}%)\n"
                f"🎯 <b>Take Profit</b>: <code>${take_profit_price:,.4f}</code> (+{self.SCALP_PROFIT_TARGET*100:.1f}%)\n"
                f"⚖️  <b>R:R</b>: 1:{self.SCALP_PROFIT_TARGET/self.SCALP_STOP_LOSS:.1f}"
            )
            await self.send_telegram_message(risk_msg, parse_mode="HTML")

            logger.info(f"✅ РИСК-МАНАДЖМЕНТ УСТАНОВЛЕН: {symbol}")
            return True

        except Exception as e:
            logger.error(f"❌ ОШИБКА РИСК-МАНАДЖМЕНТА {symbol}: {e}")
            await self.send_telegram_message(f"❌ ОШИБКА SL/TP {symbol}: {e}")
            return False

    async def _cancel_risk_orders(self, symbol: str):
        """Отмена Stop Loss и Take Profit ордеров"""
        logger.info(f"🗑️ ПРОВЕРКА ОТМЕНЫ ОРДЕРОВ: {symbol}")
        try:
            if symbol in self.stop_loss_orders:
                stop_id = self.stop_loss_orders[symbol]
                try:
                    cancel_response = self.session.cancel_order(
                        category="linear",
                        symbol=symbol,
                        orderId=stop_id
                    )
                    if cancel_response.get("retCode") == 0:
                        logger.info(f"🗑️ STOP LOSS ОТМЕНЕН: {symbol} #{stop_id}")
                    else:
                        logger.warning(f"⚠️ ОШИБКА ОТМЕНЫ SL {symbol}: {cancel_response.get('retMsg')}")
                except Exception as e:
                    logger.warning(f"⚠️ ИСКЛЮЧЕНИЕ ОТМЕНЫ SL {symbol}: {e}")
                del self.stop_loss_orders[symbol]
            else:
                logger.info(f"ℹ️ STOP LOSS НЕ НАЙДЕН: {symbol}")

            if symbol in self.take_profit_orders:
                tp_id = self.take_profit_orders[symbol]
                try:
                    cancel_response = self.session.cancel_order(
                        category="linear",
                        symbol=symbol,
                        orderId=tp_id
                    )
                    if cancel_response.get("retCode") == 0:
                        logger.info(f"🗑️ TAKE PROFIT ОТМЕНЕН: {symbol} #{tp_id}")
                    else:
                        logger.warning(f"⚠️ ОШИБКА ОТМЕНЫ TP {symbol}: {cancel_response.get('retMsg')}")
                except Exception as e:
                    logger.warning(f"⚠️ ИСКЛЮЧЕНИЕ ОТМЕНЫ TP {symbol}: {e}")
                del self.take_profit_orders[symbol]
            else:
                logger.info(f"ℹ️ TAKE PROFIT НЕ НАЙДЕН: {symbol}")

        except Exception as e:
            logger.error(f"❌ ОШИБКА ОТМЕНЫ ОРДЕРОВ {symbol}: {e}")

    async def place_scalp_order(self, symbol: str, side: str, qty: float, price: float) -> Optional[str]:
        """Размещение скальп ордера с Stop Loss и Take Profit"""
        try:
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty)
            }

            logger.info(f"🚀 ОРДЕР: {symbol} {side} | {qty:.6f} @ ${price:,.4f}")

            response = self.session.place_order(**order_params)

            if response.get("retCode") != 0:
                error_msg = f"❌ {symbol} ОШИБКА: {response.get('retMsg')} (#{response.get('retCode')})"
                logger.error(error_msg)
                await self.send_telegram_message(error_msg)
                return None

            order_id = response["result"]["orderId"]
            self.successful_signals += 1

            logger.info(f"✅ ОСНОВНОЙ ОРДЕР #{order_id} | {symbol} {side}")

            risk_success = await self._set_scalp_risk_management(symbol, side, qty, price)
            if not risk_success:
                logger.error(f"❌ ОШИБКА РИСК-МАНАДЖМЕНТА {symbol} - ОТКАТ ПОЗИЦИИ")
                try:
                    close_side = "Sell" if side == "Buy" else "Buy"
                    close_params = {
                        "category": "linear",
                        "symbol": symbol,
                        "side": close_side,
                        "orderType": "Market",
                        "qty": str(qty),
                        "reduceOnly": True
                    }
                    self.session.place_order(**close_params)
                    logger.info(f"🔄 ОТКАТ ПОЗИЦИИ {symbol}")
                except Exception as e:
                    logger.error(f"❌ ОШИБКА ОТКАТА {symbol}: {e}")
                return None

            open_time = datetime.now()
            entry_value = qty * price
            self.active_scalp_positions[symbol] = {
                "order_id": order_id,
                "side": side,
                "qty": qty,
                "entry_price": price,
                "entry_value": entry_value,
                "open_time": open_time,
                "high_watermark": price,
                "low_watermark": price,
                "rsi_at_open": self.get_rsi(symbol),
                "macd_at_open": self.calculate_macd(symbol),
                "stop_loss_set": True
            }

            logger.info(f"🕒 УСТАНОВЛЕНО OPEN_TIME: {symbol} | {open_time.strftime('%Y-%m-%d %H:%M:%S')} | Entry Value: ${entry_value:.2f}")

            message = (
                f"⚡ <b>{symbol}</b> {side} ОТКРЫТА!\n\n"
                f"💰 <code>{qty:.6f}</code> @ <code>${price:,.4f}</code>\n"
                f"📊 Размер: <b>{entry_value:.2f} USDT</b>\n"
                f"🛡️ <b>SL:</b> -{self.SCALP_STOP_LOSS*100:.1f}%\n"
                f"🎯 <b>TP:</b> +{self.SCALP_PROFIT_TARGET*100:.1f}%\n"
                f"⏰ <code>{open_time.strftime('%H:%M:%S')}</code>"
            )
            await self.send_telegram_message(message, parse_mode="HTML")

            logger.info(f"✅ ПОЗИЦИЯ С РИСКАМИ ОТКРЫТА: {symbol}")
            return order_id

        except Exception as e:
            logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА {symbol}: {e}")
            await self.send_telegram_message(f"💥 ОШИБКА ОТКРЫТИЯ {symbol}: {e}")
            return None

    async def _close_scalp_position(self, symbol: str, close_reason: str = "Manual") -> bool:
        """Закрытие скальп позиции с отменой риск-ордеров"""
        try:
            if symbol not in self.active_scalp_positions:
                logger.info(f"ℹ️ {symbol}: Позиция не найдена")
                await self._cancel_risk_orders(symbol)
                return True

            logger.info(f"🔒 ЗАКРЫТИЕ ПОЗИЦИИ: {symbol} | Причина: {close_reason}")

            position = self.active_scalp_positions[symbol]
            side = position["side"]
            qty = position["qty"]
            entry_price = position["entry_price"]
            entry_value = position.get("entry_value", qty * entry_price)
            open_time = position["open_time"]

            # Проверка существования позиции на бирже
            position_response = self.session.get_positions(category="linear", symbol=symbol)
            position_exists = False
            if position_response.get("retCode") == 0:
                position_list = position_response["result"]["list"]
                position_exists = any(
                    pos["symbol"] == symbol and pos["side"] == side and float(pos["size"]) > 0
                    for pos in position_list
                )

            # Отмена SL/TP ордеров
            await self._cancel_risk_orders(symbol)

            # Если позиция закрыта на бирже
            if not position_exists:
                logger.info(f"ℹ️ {symbol}: Позиция уже закрыта на бирже")
                exit_price = self.get_current_price(symbol)
                duration = (datetime.now() - open_time).total_seconds() / 60

                if exit_price:
                    exit_value = qty * exit_price
                    pnl_usd = exit_value - entry_value if side == "Buy" else entry_value - exit_value
                    pnl_percent = (pnl_usd / entry_value) * 100 if entry_value > 0 else 0
                    status_emoji = "🟢 ПРИБЫЛЬ" if pnl_usd > 0 else "🔴 УБЫТОК"
                    profit_color = "🟢" if pnl_usd > 0 else "🔴"

                    message = (
                        f"🔒 <b>{symbol}</b> {side} ЗАКРЫТА\n\n"
                        f"⏱️ <b>Длительность:</b> {duration:.1f} мин\n"
                        f"{profit_color} <b>P&L:</b> {pnl_usd:+.3f} USDT\n"
                        f"📊 <b>{pnl_percent:+.2f}%</b>\n"
                        f"📝 <i>{close_reason}</i>\n"
                        f"{status_emoji}"
                    )
                    logger.info(f"📊 {symbol} | {pnl_usd:+.3f} USDT ({pnl_percent:+.2f}%) | {duration:.1f}м | {close_reason}")
                else:
                    message = f"🔒 <b>{symbol}</b> закрыта | ⏱️ {duration:.1f} мин | {close_reason}"
                    logger.info(f"📊 {symbol} закрыта | {duration:.1f}м | {close_reason}")

                await self.send_telegram_message(message, parse_mode="HTML")
                del self.active_scalp_positions[symbol]
                return True

            # Закрытие позиции
            close_side = "Sell" if side == "Buy" else "Buy"
            close_params = {
                "category": "linear",
                "symbol": symbol,
                "side": close_side,
                "orderType": "Market",
                "qty": str(qty),
                "reduceOnly": True
            }

            logger.info(f"🔄 РАЗМЕЩАЕМ ОРДЕР ЗАКРЫТИЯ: {symbol} {close_side} | {qty:.6f}")
            close_response = self.session.place_order(**close_params)

            if close_response.get("retCode") == 0:
                # Получаем реальную цену исполнения
                exit_price = float(close_response["result"].get("avgPrice", self.get_current_price(symbol)))
                duration = (datetime.now() - open_time).total_seconds() / 60

                if exit_price:
                    exit_value = qty * exit_price
                    pnl_usd = exit_value - entry_value if side == "Buy" else entry_value - exit_value
                    pnl_percent = (pnl_usd / entry_value) * 100 if entry_value > 0 else 0
                    status_emoji = "🟢 ПРИБЫЛЬ" if pnl_usd > 0 else "🔴 УБЫТОК"
                    profit_color = "🟢" if pnl_usd > 0 else "🔴"

                    message = (
                        f"🔒 <b>{symbol}</b> {side} ЗАКРЫТА\n\n"
                        f"⏱️ <b>Длительность:</b> {duration:.1f} мин\n"
                        f"{profit_color} <b>P&L:</b> {pnl_usd:+.3f} USDT\n"
                        f"📊 <b>{pnl_percent:+.2f}%</b>\n"
                        f"📝 <i>{close_reason}</i>\n"
                        f"{status_emoji}"
                    )
                    logger.info(f"📊 {symbol} | {pnl_usd:+.3f} USDT ({pnl_percent:+.2f}%) | {duration:.1f}м | {close_reason}")
                else:
                    message = f"🔒 <b>{symbol}</b> закрыта | ⏱️ {duration:.1f} мин | {close_reason}"
                    logger.info(f"📊 {symbol} закрыта | {duration:.1f}м | {close_reason}")

                await self.send_telegram_message(message, parse_mode="HTML")
                del self.active_scalp_positions[symbol]
                return True
            else:
                logger.error(f"❌ ОШИБКА ЗАКРЫТИЯ {symbol}: {close_response.get('retMsg')} (#{close_response.get('retCode')})")
                return False

        except Exception as e:
            logger.error(f"❌ ИСКЛЮЧЕНИЕ ЗАКРЫТИЯ {symbol}: {e}")
            return False

    async def _manage_scalp_position(self, symbol: str):
        """Управление скальп позицией (Trailing Stop и проверка статуса)"""
        if symbol not in self.active_scalp_positions:
            return

        try:
            position = self.active_scalp_positions[symbol]
            current_price = self.get_current_price(symbol)
            if not current_price:
                logger.warning(f"⚠️ {symbol}: Не удалось получить текущую цену")
                return

            entry_price = position["entry_price"]
            side = position["side"]
            open_time = position["open_time"]
            duration = (datetime.now() - open_time).total_seconds() / 60

            logger.info(f"⏰ Проверка позиции {symbol}: Открыта {open_time.strftime('%Y-%m-%d %H:%M:%S')} | Длительность: {duration:.1f}м")

            position_response = self.session.get_positions(category="linear", symbol=symbol)
            if position_response.get("retCode") == 0:
                position_list = position_response["result"]["list"]
                position_exists = any(
                    pos["symbol"] == symbol and pos["side"] == side and float(pos["size"]) > 0
                    for pos in position_list
                )
                if not position_exists:
                    logger.info(f"🔍 ПОЗИЦИЯ {symbol} ЗАКРЫТА НА БИРЖЕ")
                    await self._close_scalp_position(symbol, "Closed on Exchange")
                    return

            if duration > self.SCALP_TIMEOUT_MINUTES:
                logger.info(f"⏰ ТАЙМАУТ {symbol}: {duration:.1f}м > {self.SCALP_TIMEOUT_MINUTES}м")
                await self._close_scalp_position(symbol, "Timeout")
                return

            if side == "Buy":
                position["high_watermark"] = max(position["high_watermark"], current_price)
            else:
                position["low_watermark"] = min(position["low_watermark"], current_price)

            should_close = False
            close_reason = ""

            if side == "Buy":
                if position["high_watermark"] > entry_price * (1 + self.SCALP_PROFIT_TARGET):
                    trail_stop = position["high_watermark"] * (1 - self.SCALP_TRAILING_STOP)
                    if current_price <= trail_stop:
                        should_close = True
                        close_reason = f"Trailing Stop {self.SCALP_TRAILING_STOP*100:.1f}%"
            else:
                if position["low_watermark"] < entry_price * (1 - self.SCALP_PROFIT_TARGET):
                    trail_stop = position["low_watermark"] * (1 + self.SCALP_TRAILING_STOP)
                    if current_price >= trail_stop:
                        should_close = True
                        close_reason = f"Trailing Stop {self.SCALP_TRAILING_STOP*100:.1f}%"

            if should_close:
                logger.info(f"🎯 TRAILING STOP {symbol}: {close_reason}")
                await self._close_scalp_position(symbol, close_reason)

        except Exception as e:
            logger.error(f"❌ ОШИБКА УПРАВЛЕНИЯ {symbol}: {e}")

    async def check_scalp_signals(self):
        """Проверка скальп сигналов с RSI и MACD"""
        if not self.running or self.BOT_MODE != "scalping":
            return

        current_time = time.time()
        if current_time - self.last_scalp_check < self.SCALP_CHECK_INTERVAL:
            return

        self.signal_checks += 1
        self.last_scalp_check = current_time

        timestamp = datetime.now().strftime('%H:%M:%S')
        active_count = len(self.active_scalp_positions)

        logger.info(f"🔍 === ПРОВЕРКА #{self.signal_checks} | {timestamp} | Активных: {active_count} ===")

        available = self.get_available_balance(self.STABLE)
        balance_str = f"{available:.2f}" if available is not None else "N/A"
        logger.info(f"💰 БАЛАНС: {balance_str} USDT")

        if available is None or available < self.SCALP_POSITION_SIZE:
            logger.warning(f"⚠️ БАЛАНС НИЗКИЙ: {balance_str}")
            return

        if active_count >= self.SCALP_MAX_POSITIONS:
            logger.info(f"⚠️ ЛИМИТ ПОЗИЦИЙ: {active_count}/{self.SCALP_MAX_POSITIONS}")
            for symbol in list(self.active_scalp_positions.keys()):
                await self._manage_scalp_position(symbol)
            return

        for symbol in list(self.active_scalp_positions.keys()):
            await self._manage_scalp_position(symbol)

        logger.info(f"📊 АНАЛИЗ ПАР: {', '.join(self.SCALP_SYMBOLS)}")
        signals_found = 0

        for i, symbol in enumerate(self.SCALP_SYMBOLS, 1):
            symbol_positions = sum(1 for pos_symbol in self.active_scalp_positions if pos_symbol == symbol)
            if symbol_positions >= self.MAX_POSITIONS_PER_SYMBOL:
                logger.info(f"  {i}. {symbol} - Лимит для пары: {symbol_positions}/{self.MAX_POSITIONS_PER_SYMBOL}")
                continue

            logger.info(f"  {i}. 📊 {symbol} - Анализ...")

            if not self.is_symbol_valid(symbol, "linear"):
                logger.info(f"  ❌ {symbol} - Недоступен")
                continue

            rsi = self.get_rsi(symbol)
            if rsi is None:
                logger.info(f"  ⏭️ {symbol} - Нет RSI")
                continue

            macd_data = self.calculate_macd(symbol, self.MACD_FAST, self.MACD_SLOW, self.MACD_SIGNAL)
            if macd_data is None:
                logger.info(f"  ⏭️ {symbol} - Нет MACD")
                continue

            price = self.get_current_price(symbol)
            if not price:
                logger.info(f"  ⏭️ {symbol} - Нет цены")
                continue

            volume_info = self.get_volume_info(symbol)
            volume_mult = volume_info["multiplier"] if volume_info else 0

            macd = macd_data["macd"]
            signal_line = macd_data["signal"]
            histogram = macd_data["histogram"]

            logger.info(f"  📈 {symbol} | RSI: {rsi:.1f} | MACD: {macd:.4f} | Signal: {signal_line:.4f} | Vol: {volume_mult:.1f}x | ${price:,.4f}")

            signal = None
            signal_strength = 0

            if rsi < self.SCALP_RSI_OVERSOLD and macd > signal_line and histogram > self.MACD_THRESHOLD:
                signal = "Buy"
                signal_strength = (self.SCALP_RSI_OVERSOLD - rsi) / 10 + histogram
                signals_found += 1
                logger.info(f"  🟢 СИГНАЛ {signal} | RSI: {rsi:.1f} | MACD: {macd:.4f} > Signal: {signal_line:.4f} | Сила: {signal_strength:.2f}")

                if signal_strength >= 0.5 and volume_mult >= self.SCALP_VOLUME_MULTIPLIER:
                    logger.info(f"  🎯 СИЛЬНЫЙ СИГНАЛ! Открываем {symbol}")
                    qty = self.calculate_scalp_qty(symbol, self.SCALP_POSITION_SIZE)
                    if qty:
                        order_id = await self.place_scalp_order(symbol, signal, qty, price)
                        if order_id:
                            await asyncio.sleep(5)
                            return
                    else:
                        logger.warning(f"  ⚠️ {symbol} - Ошибка расчета qty")
                else:
                    reason = "слабый сигнал" if signal_strength < 0.5 else "низкий объем"
                    logger.info(f"  ⏳ {symbol} - {reason} (сила: {signal_strength:.2f}, vol: {volume_mult:.1f}x)")

            elif rsi > self.SCALP_RSI_OVERBOUGHT and macd < signal_line and histogram < -self.MACD_THRESHOLD:
                signal = "Sell"
                signal_strength = (rsi - self.SCALP_RSI_OVERBOUGHT) / 10 + abs(histogram)
                signals_found += 1
                logger.info(f"  🔴 СИГНАЛ {signal} | RSI: {rsi:.1f} | MACD: {macd:.4f} < Signal: {signal_line:.4f} | Сила: {signal_strength:.2f}")

                if signal_strength >= 0.5 and volume_mult >= self.SCALP_VOLUME_MULTIPLIER:
                    logger.info(f"  🎯 СИЛЬНЫЙ СИГНАЛ! Открываем {symbol}")
                    qty = self.calculate_scalp_qty(symbol, self.SCALP_POSITION_SIZE)
                    if qty:
                        order_id = await self.place_scalp_order(symbol, signal, qty, price)
                        if order_id:
                            await asyncio.sleep(5)
                            return
                    else:
                        logger.warning(f"  ⚠️ {symbol} - Ошибка расчета qty")
                else:
                    reason = "слабый сигнал" if signal_strength < 0.5 else "низкий объем"
                    logger.info(f"  ⏳ {symbol} - {reason} (сила: {signal_strength:.2f}, vol: {volume_mult:.1f}x)")

            else:
                logger.info(f"  ➡️ {symbol} - Норма (RSI {rsi:.1f}, MACD {macd:.4f})")

        success_rate = (self.successful_signals / max(self.signal_checks, 1) * 100)
        logger.info(f"📋 === ИТОГО #{self.signal_checks} ===")
        logger.info(f"🎯 Сигналов: {signals_found} | Сделок: {self.successful_signals} | Успешность: {success_rate:.1f}%")
        logger.info(f"🔒 Позиций: {len(self.active_scalp_positions)} | SL ордеров: {len(self.stop_loss_orders)}")
        logger.info(f"⏰ Следующая проверка: +{self.SCALP_CHECK_INTERVAL}с")
        logger.info("=" * 60)

    async def _sync_positions(self):
        """Синхронизация active_scalp_positions с биржей при старте"""
        logger.info("🔄 СИНХРОНИЗАЦИЯ ПОЗИЦИЙ...")
        try:
            for symbol in self.SCALP_SYMBOLS:
                position_response = self.session.get_positions(category="linear", symbol=symbol)
                if position_response.get("retCode") != 0:
                    logger.warning(f"⚠️ Ошибка проверки позиций {symbol}: {position_response.get('retMsg')}")
                    continue

                position_list = position_response["result"]["list"]
                position_exists = any(float(pos["size"]) > 0 for pos in position_list)

                if not position_exists and symbol in self.active_scalp_positions:
                    logger.info(f"🗑️ УДАЛЕНИЕ УСТАРЕВШЕЙ ПОЗИЦИИ: {symbol}")
                    await self._close_scalp_position(symbol, "Sync: Closed on Exchange")
                elif position_exists and symbol not in self.active_scalp_positions:
                    logger.warning(f"⚠️ Обнаружена позиция на бирже {symbol}, но не в active_scalp_positions")
        except Exception as e:
            logger.error(f"❌ ОШИБКА СИНХРОНИЗАЦИИ: {e}")

    async def _signal_handler(self, signum, frame):
        logger.info(f"🛑 СИГНАЛ {signum}")
        self.running = False
        asyncio.create_task(self._cleanup())

    async def _cleanup(self):
        logger.info("🔄 ОЧИСТКА...")
        for symbol in list(self.stop_loss_orders.keys()):
            await self._cancel_risk_orders(symbol)
        for symbol in list(self.active_scalp_positions.keys()):
            await self._close_scalp_position(symbol, "Shutdown")
        logger.info("✅ ОЧИСТКА ЗАВЕРШЕНА")
        sys.exit(0)

    async def send_telegram_message(self, message: str, parse_mode: str = None):
        try:
            await self.bot.send_message(
                chat_id=self.CHAT_ID,
                text=message,
                parse_mode=parse_mode
            )
            logger.info(f"📱 ОТПРАВЛЕНО: {message[:50]}...")
        except Exception as e:
            logger.error(f"❌ TELEGRAM: {e}")

    async def main_loop(self):
        logger.info("🔄 === ОСНОВНОЙ ЦИКЛ ===")
        consecutive_errors = 0

        while self.running:
            try:
                consecutive_errors = 0
                logger.info(f"🔄 ИТЕРАЦИЯ #{self.signal_checks + 1}")

                if self.BOT_MODE == "scalping":
                    logger.info("⚡ СКАЛЬПИНГ АКТИВЕН")
                    await self.check_scalp_signals()
                    logger.info(f"😴 ПАУЗА {self.SCALP_CHECK_INTERVAL}с")
                    await asyncio.sleep(self.SCALP_CHECK_INTERVAL)
                else:
                    logger.info("💤 FUNDING РЕЖИМ")
                    await asyncio.sleep(self.CHECK_INTERVAL)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"💥 ОШИБКА #{consecutive_errors}: {e}")
                if consecutive_errors >= 3:
                    await self.send_telegram_message(f"💥 ОШИБКА ЦИКЛА: {e}")
                await asyncio.sleep(30)

    async def run(self):
        mode_name = "Scalping" if self.BOT_MODE == "scalping" else "Funding"

        try:
            logger.info(f"🚀 === {mode_name} BOT v2.7 ===")

            # Синхронизация позиций при старте
            await self._sync_positions()

            available = self.get_available_balance(self.STABLE)
            balance_display = f"{available:.2f}" if available is not None else "N/A"

            startup_message = (
                f"🤖 <b>Bybit {mode_name} Bot v2.7</b>\n\n"
                f"💰 Баланс: <code>{balance_display}</code>\n"
                f"📈 Пары: <code>{', '.join(self.SCALP_SYMBOLS)}</code>\n"
                f"⚡ Интервал: <b>{self.SCALP_CHECK_INTERVAL}с</b>\n"
                f"🛡️ <b>SL:</b> <code>{self.SCALP_STOP_LOSS*100:.1f}%</code>\n"
                f"🎯 <b>TP:</b> <code>{self.SCALP_PROFIT_TARGET*100:.1f}%</code>\n"
                f"📊 <b>MACD:</b> <code>{self.MACD_FAST}/{self.MACD_SLOW}/{self.MACD_SIGNAL}</code>\n"
                f"🔒 <b>Макс. позиций:</b> {self.SCALP_MAX_POSITIONS}\n\n"
                f"🚀 <b>ЗАПУЩЕН С STOP LOSS!</b>"
            )

            await self.send_telegram_message(startup_message, parse_mode="HTML")
            logger.info("📱 СТАРТАП ОТПРАВЛЕН")

            if self.BOT_MODE == "scalping":
                await self.send_telegram_message(
                    f"⚡ <b>СКАЛЬПИНГ С STOP LOSS АКТИВЕН</b>\n\n"
                    f"🛡️ <b>Автоматическая защита:</b>\n"
                    f"• Stop Loss: -{self.SCALP_STOP_LOSS*100:.1f}%\n"
                    f"• Take Profit: +{self.SCALP_PROFIT_TARGET*100:.1f}%\n"
                    f"• Trailing Stop: {self.SCALP_TRAILING_STOP*100:.1f}%\n"
                    f"• MACD: {self.MACD_FAST}/{self.MACD_SLOW}/{self.MACD_SIGNAL}\n\n"
                    f"🔍 <b>Поиск каждые {self.SCALP_CHECK_INTERVAL}с</b>\n"
                    f"📊 <b>Полная защита позиций</b>\n\n"
                    f"🎯 <b>ГОТОВ К БЕЗОПАСНОЙ ТОРГОВЛЕ!</b> 🛡️",
                    parse_mode="HTML"
                )
                logger.info("⚡ УВЕДОМЛЕНИЕ С STOP LOSS")

            await self.main_loop()

        except Exception as e:
            logger.critical(f"💥 ФАТАЛЬНАЯ ОШИБКА: {e}")
            await self.send_telegram_message(f"💥 <b>КРИТИЧЕСКАЯ ОШИБКА</b>\n<code>{e}</code>", parse_mode="HTML")
        finally:
            await self._cleanup()

async def main():
    logger.info("🎯 === MAIN ЗАПУЩЕН ===")
    try:
        bot = BybitFundingBot()
        logger.info("✅ Бот создан")
        await bot.run()
    except Exception as e:
        logger.error(f"💥 MAIN ОШИБКА: {e}")
        print(f"💥 Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("🚀 СКРИПТ СТАРТ")
    asyncio.run(main())