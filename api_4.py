import os
import asyncio
import aiohttp
import pandas as pd
import statsmodels.api as sm
import logging


class PriceMonitor:
    """
    Класс, который осуществляет мониторинг цены Ethereum по отношению к цене Bitcoin и проверяет изменение цены.

    Атрибуты
    ----------
    analysis_period : int
        Период анализа цены, заданный в минутах.
    price_change_threshold : float
        Порог изменения цены, заданный в процентах.

    Методы
    -------
    __init__(self, analysis_period: int, price_change_threshold: float) -> None:
        Инициализирует экземпляр класса PriceMonitor.

    calculate_price_change(self, current_price: float, previous_price: float) -> float:
        Вычисляет процентное изменение цены.

    async def get_price_data(self, session: aiohttp.ClientSession, endpoint: str) -> pd.DataFrame:
        Получает данные о цене с Binance API и преобразует их в pandas DataFrame.

    async def monitor_price_change(self) -> None:
        Осуществляет мониторинг изменения цены Ethereum и выводит информацию об изменении цены.
    """
    def __init__(self, analysis_period: int, price_change_threshold: float) -> None:
        """
        Инициализирует экземпляр класса PriceMonitor.

        Параметры
        ----------
        analysis_period : int
            Период анализа цены, заданный в минутах.
        price_change_threshold : float
            Порог изменения цены, заданный в процентах.
        """

        self.analysis_period = analysis_period
        self.price_change_threshold = price_change_threshold

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.warning_logger = logging.getLogger('warning')
        self.warning_logger.setLevel(logging.WARNING)
        self.error_logger = logging.getLogger('error')
        self.error_logger.setLevel(logging.ERROR)

    def calculate_price_change(self, current_price: float, previous_price: float) -> int | float:
        """
        Вычисляет процентное изменение цены.

        Параметры
        ----------
        current_price : float
            Текущая цена.
        previous_price : float
            Предыдущая цена.

        Возврат
        -------
        float | int
            Процентное изменение цены.

        """
        if previous_price:
            return abs(current_price - previous_price) / previous_price * 100
        else:
            return 0

    async def get_price_data(self, session: aiohttp.ClientSession, endpoint: str) -> pd.DataFrame:
        """
        Получает данные о цене с Binance API и преобразует их в объект pandas DataFrame.

        Параметры
        ----------
        session : aiohttp.ClientSession
            Сессия клиента для обращения к API Binance.
        endpoint : str
            Адрес эндпоинта для запроса данных.

        Возврат
        -------
        pd.DataFrame
            DataFrame с данными о цене.

        """
        try:
            async with session.get(endpoint) as response:
                response.raise_for_status()
                data = await response.json()
                df = pd.DataFrame(data)
                df = df.iloc[:, :6]
                df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                df['close'] = pd.to_numeric(df['close'])
                # self.logger.info(f"Получено {len(df)} строк данных из {endpoint}")
                return df
        except Exception as e:
            self.warning_logger.warning(f"Ошибка при получении данных из {endpoint}: {e}")

    async def monitor_price_change(self):
        """
        Осуществляет мониторинг изменения цены Ethereum и выводит информацию об изменении цены.
        """
        while True:
            async with aiohttp.ClientSession() as session:
                eth_endpoint = 'https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=1m'
                btc_endpoint = 'https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m'
                eth_data_task = asyncio.create_task(self.get_price_data(session, eth_endpoint))
                btc_data_task = asyncio.create_task(self.get_price_data(session, btc_endpoint))

                await asyncio.gather(eth_data_task, btc_data_task)

                eth_data = eth_data_task.result()
                btc_data = btc_data_task.result()

                if eth_data is not None and btc_data is not None:
                    # получаем данные за последний месяц
                    eth_data = eth_data[
                        eth_data['datetime'] >= pd.Timestamp.now() - pd.Timedelta(minutes=self.analysis_period)]
                    btc_data = btc_data[
                        btc_data['datetime'] >= pd.Timestamp.now() - pd.Timedelta(minutes=self.analysis_period)]

                    # создаём модельку регресс-анализа
                    x = btc_data['close']
                    y = eth_data['close']
                    X = sm.add_constant(x)
                    model = sm.OLS(y, X).fit()


                    # делаем предсказание цены этериума по отношению текущей цены биткоина
                    current_btc_price = btc_data['close'].iloc[-1]
                    predicted_eth_price = model.predict([1, current_btc_price])[0]

                    # считаем изменение этериума
                    current_eth_price = eth_data['close'].iloc[-1]
                    price_change = self.calculate_price_change(current_eth_price, eth_data['close'].iloc[-2])

                    self.logger.info(f"Текущая цена ETH: {current_eth_price:.3f}")
                    self.logger.info(
                        f"Предсказанная цена ETH при текущей цене BTC ({current_btc_price:.3f}): {predicted_eth_price:.3f}")
                    self.logger.info(f"Изменение цены: {price_change:.3f}%")

                    # проверка порогового значения в 1 %
                    if price_change > self.price_change_threshold:
                        self.warning_logger.warning("Изменение цены превышает порог в 1%!")

            await asyncio.sleep(60)


if __name__ == '__main__':
    PERIOD = 43_200  # месяц в минутах
    PRICE_CHANGE_THRESHOLD = 1
    pm = PriceMonitor(PERIOD, PRICE_CHANGE_THRESHOLD)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(pm.monitor_price_change())
    loop.run_forever()
