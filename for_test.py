from pybit.unified_trading import HTTP
session = HTTP(api_key="tBHpsOXb36gJofHkgF", api_secret="lazdaPLNjW78Jg6VWB6psa2FuQjLypKgDWlL", testnet=False)
print(session.get_wallet_balance(accountType="UNIFIED", coin="USDT"))