from global_utils.multi_proc.base_parallel_call import BaseParallelCall
from global_utils.multi_proc.parallel_call_helper import parallel_call
from global_utils.currencyDataProvider import currencyDataProvider
from global_utils.customizedCalendar import customizedCalendar
from global_utils.tradingDayTracker import tradingDayTracker
from global_utils.versionController import versionController
from global_utils.notionalTracker import notionalTracker
from global_utils.pnlCalculator import pnlCalculator
from global_utils.dataCollector import dataCollector
from global_utils.tickerMapper import tickerMapper
from global_utils.dataLoader import dataLoader
from global_utils.backupFile import backupFile
from global_utils.dataFeeder import dataFeeder
from global_utils.fileWriter import fileWriter
from global_utils.universe import universe
from global_utils.sector import sector
from global_config.configReader import configReader
from global_data_updater.sftpHandler import sftpHandler
from mosek.fusion import Model, Domain, ObjectiveSense, Expr, SolutionStatus
from mosek_service.client.mosek_api_client import MosekApiClient
import multiprocessing as mp
from copy import deepcopy
import datetime as dt
import pandas as pd
import numpy as np
import pickle
import time
import math
import os
from tenacity import *

class positionGenerator(BaseParallelCall):
    staticConfig = configReader.readConfig('config/config.xml')
    alphaSkeleton = staticConfig.alpha.contents[0] + '/%s/%s/%s_%s_%s.csv'
    betaSkeleton = staticConfig.beta.contents[0] + '/%s/beta_%s_%s.csv'
    precloseDataSkeleton = staticConfig.preclose.contents[0] + '/%s/preclose_%s_%s.csv'
    closeDataSkeleton = staticConfig.ciqmarkit.contents[0] + '/%s/a/a_%s_%s.csv'
    bbgCorporateActionSkeleton = staticConfig.bbg.contents[0] + '/corporateaction/%s/corporateaction_%s_%s.csv'

    firstUnitNotionalDate = dt.datetime(2007, 5, 8)
    firstPositionDate = dt.datetime(2007, 11, 8)
    firstPrecloseDate = dt.datetime(2018, 5, 16)

    alphaCol = 'ALP_ZSC_BL'
    volWin = 240
    volTarget = 0.1
    singleLimit = 0.02
    sectorLimit = 0.15
    volumeLimit = 0.01
    volumeWin = 20
    volumeWinExtended = 63
    adtvFilterWin = 22
    smoothingWin = 5
    countryScalingCap = 3
    apacScalingCap = 2
    defaultShortSellRateForUnconstraint = 3.5
    minMktCapFloatUSDIn = 1.2e8
    minMktCapFloatUSDOut = 1e8
    minUnitPriceUSDIn = 0.2
    minUnitPriceUSDOut = 0.1
    minADTVUSDIn = 2.2e5
    minADTVUSDOut = 1.8e5
    tcThresholdMultiple = 0.2
    shortSellRateCapMapping = {
        'AUS': 6,
        'HKG': 6,
        'KOR': 6,
        'SIN': 6,
        'TAI': 6,
    }
    defaultRoundLotMapping = {
        'AUS': 1,
        'HKG': 2000,
        'KOR': 1,
        'SIN': 100,
        'TAI': 1000,
    }

    btFields = ['BC', 'AC']
    fillTilts = []

    debugMode = False

    def __init__(
            self,
            sDate,
            eDate,
            broker,
            countries=None,
            sig=None,
            executionType='CLOSE',
            beforeMarketOpen=False,
            imposeResponseCut=True,
            shortSellUnconstraintMode=False,
            shortSellInventoryDate=None,
            includePrevinventoryShortSell=True,
            multiProcessMode=True,
            backupMode=True,
            liveMode=True,
            backtestList=None,
            useID=False,
            useCache=False,
    ):
        print(dt.datetime.now(), 'deployment check 2020-10-12')
        self.sDate = sDate
        self.eDate = eDate
        self.broker = broker
        if sig is None:
            self.sig = str(self.staticConfig.signal.contents[0])
        else:
            self.sig = sig
        self.beforeMarketOpen = beforeMarketOpen
        self.multiProcessMode = multiProcessMode
        self.imposeResponseCut = imposeResponseCut
        self.executionType = executionType.upper()
        self.shortSellInventoryDate = shortSellInventoryDate
        self.shortSellUnconstraintMode = shortSellUnconstraintMode
        self.includePrevinventoryShortSell = includePrevinventoryShortSell
        self.backupMode = backupMode
        self.liveMode = liveMode
        self.useCache = useCache
        if backtestList is None:
            self.backtestList = ['',]
        else:
            self.backtestList = backtestList
        self.cc = customizedCalendar()
        if countries is None or countries == ['APAC']:
            self.countries = self.staticConfig.countriestrade.contents[0].split(';')
        else:
            self.countries = countries
        if self.backtestList is None or len(self.backtestList) == 0:
            self.backtestList = ['']
        # self.tickerType = 'CIQID'
        self.tickerType = 'id'
        if useID:
            self.tickerType = 'id'
        self.tdt = tradingDayTracker()
        launchDate = dt.datetime.strptime(self.staticConfig.find('ms').find('launchdate').contents[0], '%Y/%m/%d')
        self.launchDateDict = dict([(country, self.getNextRebalanceDayFrom(launchDate, country)) for country in self.countries])
        self.estuExcludedCIQIDsByCountry = dict([(x.split(',')[0], x.split(',')[1:]) for x in self.staticConfig.estuexcludedtickers.contents[0].split(';')])
        vc = versionController(self.broker)
        versionLaunchDate = vc.getCurrentVersionLaunchDate()
        self.cdp = currencyDataProvider(useCache=self.useCache)
        self.nt = notionalTracker(self.broker)
        self.dc = dataCollector()
        self.tm = tickerMapper()
        self.sh = sftpHandler(
            self.broker,
            countries=self.countries,
            beforeMarketOpen=self.beforeMarketOpen,
        )
        self.dl = dataLoader()
        self.sector = sector()
        self.univ = universe()
        self.bf = backupFile()
        self.firstDateCache = {}
        self.dataFeeders = {}
        if not self.useCache:
            self.resetNbRequestResponseCache()
        if countries != ['APAC']:
            self.initialize()
        self.resetRequestSettings()
        self.resetPositionRoot(str(self.staticConfig.find(self.broker.lower()).find('position').contents[0]))
        if countries == ['APAC']:
            self.countries = countries
        self.lastExceptionDate = pd.datetime(2000, 1, 1)
        self.instrumentMapping = {}
        self.pbWeight = float(self.staticConfig.find('pbweight').find(self.broker.lower()).contents[0])

    def resetPositionRoot(self, positionRoot):
        self.positionRoot = positionRoot

    def resetRequestSettings(self):
        self.adjustByShortSellResponse = False

    def setRequestSettings(self, adjustByShortSellResponse):
        self.adjustByShortSellResponse = adjustByShortSellResponse
        if self.backupMode:
            self.bf.setBackupFolder(os.path.join(self.broker.lower(), 'position'))

    def resetNbRequestResponseCache(self):
        if self.imposeResponseCut:
            self.nbRequestResponse = {}
            dates = self.cc.getDatesBetweenInclusiveCountryUnion(self.sDate, self.eDate, countries=self.countries)
            for asOfDate in dates:
                nbRequest = self.sh.countRequest(asOfDate)
                self.nbRequestResponse[asOfDate] = nbRequest

    def initialize(self):
        preSDate = self.cc.getPrevDateBeforeCountryIntersection(self.sDate, self.countries, n=self.volWin + 5 + max(4, self.volumeWin, self.volumeWinExtended))
        preEDate = self.cc.getPrevDateBeforeCountryUnion(self.eDate, self.countries)
        self.tickersByCountry = {}

        self.pnlc = pnlCalculator(
            self.broker,
            preSDate,
            preEDate,
            countries=self.countries,
            executionType=self.executionType,
            multiProcessMode=self.multiProcessMode,
            useCache=self.useCache,
        )

        self.preSDate = preSDate
        self.preEDate = preEDate

    def initializeByCountry(self, country):
        self.tickersByCountry[country] = self.univ.getUniverse(country, self.tickerType, useCache=self.useCache)

    def getPrevRebalanceDayBy(self, asOfDate, country):
        if self.tdt.isRebalanceDay(asOfDate, country):
            return asOfDate
        else:
            return self.getPrevRebalanceDayBy(self.cc.getPrevDateBefore(asOfDate, country), country)

    def getNextRebalanceDayFrom(self, asOfDate, country):
        if self.tdt.isRebalanceDay(asOfDate, country):
            return asOfDate
        else:
            return self.getNextRebalanceDayFrom(self.cc.getNextDateAfter(asOfDate, country), country)

    def loadSuspendSet(self, asOfDate, country):
        readDate = self.cc.getPrevDateBefore(asOfDate, country) if self.beforeMarketOpen else asOfDate
        # fileName = self.suspendSkeleton % (country, country, readDate.strftime('%Y%m%d'))
        # data = pd.read_csv(fileName)[self.tickerType]
        if self.useCache:
            suspendList = self.dl.loadInstrumentPeriodEventCache(readDate, country, 'SUSPEND').index.tolist()
        else:
            suspendList = self.dl.loadSuspendData(readDate, country, tickerType=self.tickerType).tolist()
        return set(suspendList)

    def getExcludeTickers(self, asOfDate, country, side):
        removalExclusionSides = {'BOTH', side}
        if self.useCache:
            removalOverwriteTickers = self.dl.loadModelPeriodEventCache(asOfDate, country, self.broker, 'OVERWRITE')
            overheatTickers = self.dl.loadModelPeriodEventCache(asOfDate, country, self.broker, 'OVERHEAT').index.tolist() if side in {'SHORT', 'BOTH'} else []
            hightouchList = self.dl.loadModelPeriodEventCache(asOfDate, country, self.broker, 'HIGHTOUCH').index.tolist()
        else:
            removalOverwriteTickers = self.dl.loadModelPeriodEventData(asOfDate, self.broker, self.tickerType, 'OVERWRITE')
            overheatTickers = self.dl.loadModelPeriodEventData(asOfDate, self.broker, self.tickerType, 'OVERHEAT').index.tolist() if side in {'SHORT', 'BOTH'} else []
            hightouchList = self.dl.loadModelPeriodEventData(asOfDate, self.broker, self.tickerType, 'HIGHTOUCH').index.tolist()
        removalOverwriteTickers = removalOverwriteTickers.loc[removalOverwriteTickers['OVERWRITE'].apply(lambda x: x in removalExclusionSides), :].index.tolist()
        estuExcludedTickers = self.estuExcludedCIQIDsByCountry[country] if country in self.estuExcludedCIQIDsByCountry else []
        estuExcludedTickers = self.instrumentMapping[country].set_index('CIQID').loc[estuExcludedTickers, self.tickerType].to_list()
        if self.debugMode:
            nanHightouchList = [pd.isnull(i) for i in hightouchList]
            if nanHightouchList:
                print(dt.datetime.now(), 'hight touch nan issue detected')
                # print(dt.datetime.now(), country, asOfDate)
                print(dt.datetime.now(), 'estuExcludedTickers')
                # print(dt.datetime.now(), estuExcludedTickers)
                hightouchList = [pd.notnull(i) for i in hightouchList]
        excludedTickers = sorted(list(set(removalOverwriteTickers).union(estuExcludedTickers).union(overheatTickers).union(hightouchList)))
        return excludedTickers

    def getAlphas(self, asOfDate, country):
        fileName = self.alphaSkeleton % (country, self.sig, self.sig, country, asOfDate.strftime('%Y%m%d'))
        alphaData = pd.read_csv(fileName, index_col=self.tickerType)
        if 'id' in alphaData.columns:
            alphaData['id'] = alphaData['id'].apply(lambda x: int(x) if pd.notnull(x) else None)
        excludedTickers = self.getExcludeTickers(asOfDate, country, 'BOTH')
        alphas = alphaData.loc[[ticker not in excludedTickers for ticker in alphaData.index], self.alphaCol]
        alphas = alphas[alphas.notnull()]
        return alphas

    def loadPortfolio(self, asOfDate, country, positionType, backtestVersion=''):
        if asOfDate < self.firstUnitNotionalDate:
            data = pd.DataFrame(columns=['USD', 'RATE'])
        else:
            fileName = os.path.join(self.positionRoot, backtestVersion, country, positionType, 'position_%s_%s_%s.csv' % (positionType, country, asOfDate.strftime('%Y%m%d')))
            data = pd.read_csv(fileName)
            if 'id' not in data.columns:
                instrumentMapping = self.instrumentMapping[country]
                data = pd.merge(data, instrumentMapping[['SEDOL', 'id']], on='SEDOL')
            data = data.set_index(self.tickerType)[['USD', 'RATE']]
            # data.index = self.dl.getOverwrittenTickers(asOfDate, 'CIQID', data.index, country=country)
        return data

    def getReevaluatedPrevPortfolio(self, asOfDate, country, positionType, updateRates=True, backtestVersion=''):
        prevDate = self.cc.getPrevDateBefore(asOfDate, country)
        prevPortfolio = self.loadPortfolio(prevDate, country, positionType, backtestVersion=backtestVersion)
        estuExcludedTickersLong = self.getExcludeTickers(asOfDate, country, 'LONG')
        estuExcludedTickersShort = self.getExcludeTickers(asOfDate, country, 'SHORT')
        prevPortfolioLong = prevPortfolio.loc[prevPortfolio['USD'] > 0, :]
        prevPortfolioShort = prevPortfolio.loc[prevPortfolio['USD'] < 0, :]
        prevPortfolioFilteredLong = prevPortfolioLong.loc[[ticker not in estuExcludedTickersLong for ticker in prevPortfolioLong.index], :]
        prevPortfolioFilteredShort = prevPortfolioShort.loc[[ticker not in estuExcludedTickersShort for ticker in prevPortfolioShort.index], :]
        prevPortfolioFiltered = pd.concat([prevPortfolioFilteredLong, prevPortfolioFilteredShort], axis=0)
        reevaluatedPrevPortfolio = self.pnlc.reevaluatedPortfolio(asOfDate, country, prevPortfolioFiltered)
        if updateRates:
            shortSellData = self.getShortSellQuantityData(asOfDate, country)
            tickersRateUpdate = list(set(reevaluatedPrevPortfolio.index[reevaluatedPrevPortfolio['USD'] < 0]).intersection(shortSellData.index))
            reevaluatedPrevPortfolio.loc[tickersRateUpdate, 'RATE'] = shortSellData.loc[tickersRateUpdate, 'RATE']
        return reevaluatedPrevPortfolio

    def after_retry_callback(a, b, c):
        print(dt.datetime.now(), 'retry was invoked due to exception', dt.datetime.now())

    # @retry(wait=wait_random(1, 3), stop=stop_after_attempt(100), after=after_retry_callback)
    def loadRiskData(self, asOfDate, country):
        self.instrumentMapping[country] = self.dl.getInstruments(asOfDate, country, self.useCache)
        self.alphas = self.getAlphas(asOfDate, country)
        loadDate = self.dl.getMappingLoadingDate(asOfDate, 'barra')
        self.exposure = self.dl.loadBarraExposureData(loadDate, country, self.tickerType)
        #ToDo remove duplicates, need to be removed when data is fixed
        if self.debugMode:
            if not self.exposure.loc[self.exposure.index.duplicated(),:].empty:
                duplicatedIndexSet = set(self.exposure.index[self.exposure.index.duplicated()].tolist())
                duplicatedExposure = self.exposure.loc[[idx in duplicatedIndexSet for idx in self.exposure.index], :]
                print(dt.datetime.now(), 'duplicated index in Exposure detected,')
                # print(dt.datetime.now(), duplicatedExposure)
                self.exposure = self.exposure.loc[~self.exposure.index.duplicated(), :]
        self.cov = self.dl.loadBarraCovData(loadDate, country)
        if self.useCache:
            self.srisk = self.dl.loadData('SRISK', loadDate, loadDate, country)
        else:
            self.srisk = self.dl.loadBarraSRiskData(loadDate, country, self.tickerType)['SRISK']
        # ToDo remove duplicates, need to be removed when data is fixed
        if self.debugMode:
            if not self.srisk.loc[self.srisk.index.duplicated()].empty:
                duplicatedIndexSet = set(self.srisk.index[self.srisk.index.duplicated()].tolist())
                duplicatedSRisk = self.srisk.loc[[idx in duplicatedIndexSet for idx in self.srisk.index]]
                print(dt.datetime.now(), 'duplicated index in SRisk detected,')
                self.srisk = self.srisk.loc[~self.srisk.index.duplicated()]
        shortSellUSDData = self.getShortSellUSDData(asOfDate, country)
        excludedTickers = self.getExcludeTickers(asOfDate, country, 'SHORT')
        self.shortSellData = shortSellUSDData.loc[[ticker not in excludedTickers for ticker in shortSellUSDData.index], :]
        self.betas = self.getBeta(asOfDate, country, positionType='risk_apac')
        # self.sectors = self.sector.getSectors(asOfDate, country, tickerType=self.tickerType)
        spot = self.cdp.getSpot(self.cc.getPrevDateBefore(asOfDate, country), country, type='close')
        self.tradingLimitUSD = self.getPreviousMeanVolumeLC(asOfDate, country, self.alphas.index, self.volumeWin, self.volumeWinExtended).multiply(spot)

    def getTargetFileName(self, country, backtestVersion):
        fileName = os.path.join(self.positionRoot, backtestVersion, country, 'target', 'target_%s.csv' % country)
        return fileName

    def loadTarget(self, country, backtestVersion):
        fileName = self.getTargetFileName(country, backtestVersion)
        if os.path.isfile(fileName):
            target = pd.read_csv(fileName, index_col='DATE')
            target.index = [dt.datetime.strptime(x, '%Y/%m/%d') for x in target.index]
        else:
            target = pd.DataFrame()
        target.index.name = 'DATE'
        return target

    def saveTarget(self, target, country, backtestVersion):
        fileName = self.getTargetFileName(country, backtestVersion)
        data = deepcopy(target)
        data.index = [d.strftime('%Y/%m/%d') for d in data.index]
        data.index.name = 'DATE'
        fileWriter.to_csv(data, fileName)

    def saveException(self, country, backtestVersion, asOfDate, type, oldValue, newValue):
        print(dt.datetime.now(), 'Solution cannot be found on %s %s, relaxing %s limit constraint to %f' % (
                country, asOfDate.strftime('%Y-%m-%d'), type, newValue))
        fileName = os.path.join(self.positionRoot, backtestVersion, country, 'exception', 'exception_%s.csv' % country)
        if os.path.exists(fileName):
            exceptionList = pd.read_csv(fileName)
        else:
            exceptionList = pd.DataFrame()
        exceptionListNew = pd.Series(
            [asOfDate.strftime('%Y/%m/%d'), country, backtestVersion, 'Single', oldValue, newValue],
            index=['DATE', 'COUNTRY', 'Backtest Version', 'CONSTRAINT', 'OLD VALUE', 'NEW VALUE'])
        exceptionList = pd.concat([exceptionList, exceptionListNew.to_frame().T], axis=0)
        exceptionList = exceptionList.drop_duplicates()
        fileWriter.to_csv(exceptionList, fileName, index=False)

    def solvePortfolio(self, asOfDate, country, prevPortfolio, prevPortfolioSuspendTickers, tickerRates, backtestVersion=''):
        factorsWithoutAlphaViewStyle = self.dl.factorsWithoutAlphaViewStyle + self.dl.jpStyleFactors
        factorsWithoutAlphaViewIndustry = self.dl.industryFactors + self.dl.jpIndustryFactors
        marketFactors = self.dl.marketFactors
        countryFactors = self.dl.countryFactors
        currencyFactors = self.dl.currencyFactors

        notional = self.nt.getNotional(asOfDate, country, 'model')
        netNotionalBound = 0.0

        # idioRiskBoundMapping = 0.14
        # factorRiskBound = np.nan
        # idioRiskBound = idioRiskBoundMapping

        robustWeightMapping = 6
        robustWeight = robustWeightMapping

        shortSellMultiplierMapping = {
            'AUS': 1,
            'HKG': 60,
            'KOR': 100,
            'SIN': 60,
            'TAI': 130,
        }
        shortSellMultiplier = shortSellMultiplierMapping[country]

        # factorLimitMapping = np.inf
        # factorLimit = factorLimitMapping

        alphaWeightMapping = 4
        alphaWeight = alphaWeightMapping

        tcWeightMapping = 1.2
        tcWeight = tcWeightMapping

        # betaExposureLimitMapping = 0.05
        # betaExposureLimit = betaExposureLimitMapping

        if self.nt.getNotional(asOfDate, country, 'model') == self.nt.getNotional(self.cc.getPrevDateBefore(
                asOfDate, country), country, 'model'):
            volumeLimitMapping = 0.05
        else:
            volumeLimitMapping = np.inf
        volumeLimit = volumeLimitMapping * self.pbWeight

        betaVarianceLimitMapping = 0.1
        betaVarianceLimit = betaVarianceLimitMapping * np.square(self.volTarget)

        factorStyleLimitMapping = 0.1
        factorStyleLimit = pd.Series(np.sqrt(factorStyleLimitMapping) * self.volTarget, index=factorsWithoutAlphaViewStyle)
        factorStyleLimit[['DSBETA', 'RESVOL', 'LIQUIDTY']] = np.sqrt(0.05) * self.volTarget

        factorIndustryLimitMapping = 0.1
        factorIndustryLimit = factorIndustryLimitMapping * np.square(self.volTarget)

        factorMarketLimitMapping = 0.1
        factorMarketLimit = factorMarketLimitMapping * np.square(self.volTarget)

        factorCountryLimitMapping = 0.1
        factorCountryLimit = factorCountryLimitMapping * np.square(self.volTarget)

        factorCurrencyLimitMapping = 0.1
        factorCurrencyLimit = factorCurrencyLimitMapping * np.square(self.volTarget)

        factorExposureLimitMapping = np.inf
        factorExposureLimit = factorExposureLimitMapping

        overallStyleLimitMapping = 0.2
        overallStyleLimit = overallStyleLimitMapping * np.square(self.volTarget)

        overallSectorLimitMapping = 0.2
        overallSectorLimit = overallSectorLimitMapping * np.square(self.volTarget)

        overallMarketLimitMapping = 0.2
        overallMarketLimit = overallMarketLimitMapping * np.square(self.volTarget)

        overallCountryLimitMapping = 0.2
        overallCountryLimit = overallCountryLimitMapping * np.square(self.volTarget)

        overallCurrencyLimitMapping = 0.2
        overallCurrencyLimit = overallCurrencyLimitMapping * np.square(self.volTarget)

        # betaExposureLimitMapping = 0.05
        #testing using more strict constraint0.01 and 0.005
        # betaExposureLimitMapping = 0.01
        betaExposureLimitMapping = 0.005
        betaExposureLimit = betaExposureLimitMapping

        countryInflationMapping = {
            'AUS': 1.32,
            'HKG': 1.40,
            'KOR': 1.6,
            'SIN': 1.14,
            'TAI': 1.25,
        }
        countryInflationFactor = countryInflationMapping[country]

        target = self.loadTarget(country, backtestVersion)
        apacMultiplier = 1.7
        target = target.loc[target.index <= asOfDate, :]
        if target.shape[0] > 0:
            target = target.reindex(sorted(list(set(target.index).union([asOfDate]))))
            target = target.ffill()
            prevAdjFactor = target.loc[asOfDate, 'Total Adj Factor']
        else:
            prevAdjFactor = apacMultiplier
        volTarget = self.volTarget
        scalingCap = self.countryScalingCap * self.apacScalingCap

        riskTargetSmoothingMapping = 1
        riskTargetSmoothing = riskTargetSmoothingMapping

        alphas = deepcopy(self.alphas)
        exposure = deepcopy(self.exposure)
        srisk = deepcopy(self.srisk)
        cov = deepcopy(self.cov)
        shortSellData = deepcopy(self.shortSellData)
        betas = deepcopy(self.betas)
        # sectors = deepcopy(self.sectors)

        alphas = alphas[tickerRates.index]
        suspendTickers = sorted(list(prevPortfolioSuspendTickers))
        newTickers = sorted(list(set(alphas.index).difference(prevPortfolioSuspendTickers)))
        tickers = suspendTickers + newTickers
        # nonSuspendNotional = 2 * notionalMultiplier[country] * notional - prevPortfolio.loc[prevPortfolioSuspendTickers, 'USD'].abs().sum()
        alphas = alphas.reindex(tickers)
        if self.debugMode:
            #ToDo Check Data Quality
            if not betas[betas.index.duplicated()].empty:
                print(dt.datetime.now(), 'solvePortfolio in positionGenerator, duplicated index betas detected.')
                duplicatedIndexSet = set(betas.index[betas.index.duplicated()].tolist())
                duplicatedBetas = betas[[idx in duplicatedIndexSet for idx in betas.index]]
                # print(dt.datetime.now(), duplicatedBetas)
                betas = betas[~betas.index.duplicated()]
        try:
            betas = betas.reindex(tickers).fillna(1)
        except:
            print('beta issue detected', country, asOfDate)
            if not betas[betas.index.duplicated()].empty:
                print(dt.datetime.now(), 'solvePortfolio in positionGenerator, duplicated index betas detected.')
                duplicatedIndexSet = set(betas.index[betas.index.duplicated()].tolist())
                duplicatedBetas = betas[[idx in duplicatedIndexSet for idx in betas.index]]
                print('duplicatedBetas', duplicatedBetas)
                betas = betas[~betas.index.duplicated()]
        exposure = exposure.loc[tickers, self.dl.barraFactorsFull]
        invalidBarraTickers = exposure.loc[pd.isnull(exposure['BETA']), :].index.tolist()
        colList = ['BETA']
        exposure.loc[:, colList] = exposure[colList].fillna(1)
        exposure = exposure.fillna(0)
        srisk = srisk.loc[tickers].fillna(srisk.median())
        exposureFull = deepcopy(exposure)
        cov = cov.loc[self.dl.barraFactorsFull, self.dl.barraFactorsFull]
        cov = cov.fillna(0)
        covFull = deepcopy(cov)
        exposure = exposure.loc[:, self.dl.barraFactorsFull]
        validFactors = abs(exposure).sum() > 0.01
        exposure = exposure.loc[:, validFactors]
        cov = cov.loc[validFactors, validFactors]
        f = np.linalg.cholesky(cov.values)
        FU = pd.DataFrame(f, index=cov.index, columns=cov.columns).T.dot(exposure.T)
        n = alphas.shape[0]
        nSuspend = len(prevPortfolioSuspendTickers)
        # shortSellDays = (self.cc.getNextDateAfter(asOfDate, country) - asOfDate).days
        shortSellDays = 1
        shortSellData.loc[:, 'USD'] = -shortSellData['USD']
        shortSellData = shortSellData.reindex(tickers)
        shortSellData.loc[prevPortfolioSuspendTickers, 'USD'] = -np.inf
        shortSellData.loc[:, 'RATE'] = shortSellData['RATE'].mul(0.01 * shortSellDays * 252 / 360)
        shortSellData = shortSellData.fillna(0.0)
        prevPortfolio = prevPortfolio.reindex(tickers)
        prevPortfolio = prevPortfolio.fillna(0.0)
        # cost = (self.pnlc.buyCostByCountry[country] + self.pnlc.sellCostByCountry[country]) / 2 * 252
        buyExchangeFeeByCountry = self.pnlc.cst.getCost(asOfDate, self.broker, country, 'model', 'buyexchangefee')
        sellExchangeFeeByCountry = self.pnlc.cst.getCost(asOfDate, self.broker, country, 'model', 'sellexchangefee')
        commissionByCountry = self.pnlc.cst.getCost(asOfDate, self.broker, country, 'model', 'brokerage')
        slippageByCountry = self.pnlc.cst.getCost(asOfDate, self.broker, country, 'model', 'slippage')
        buyCostByCountry = buyExchangeFeeByCountry + commissionByCountry + slippageByCountry
        sellCostByCountry = sellExchangeFeeByCountry + commissionByCountry + slippageByCountry
        cost = (buyCostByCountry + sellCostByCountry) / 2 * 252
        singleLimitMapping = 0.01
        if asOfDate > self.cc.getNextDateFrom(self.firstUnitNotionalDate, country):
            singleLimit = singleLimitMapping * self.countryScalingCap * self.apacScalingCap * 2
        else:
            singleLimit = singleLimitMapping * 2

        # factorStyleLimit = max(factorStyleLimit, np.square(FU.loc[set(FU.index).intersection(
        #     factorsWithoutAlphaViewStyle), suspendTickers].dot(prevPortfolio.loc[suspendTickers, 'USD']/notional).abs().max()))
        # factorIndustryLimit = max(factorIndustryLimit, np.square(FU.loc[set(FU.index).intersection(
        #     factorsWithoutAlphaViewIndustry), suspendTickers].dot(prevPortfolio.loc[suspendTickers, 'USD']/notional).abs().max()))
        # factorCountryLimit = max(factorCountryLimit, np.square(FU.loc[set(FU.index).intersection(
        #     countryFactors), suspendTickers].dot(prevPortfolio.loc[suspendTickers, 'USD']/notional).abs().max()))
        # factorMarketLimit = max(factorMarketLimit, np.square(FU.loc[set(FU.index).intersection(
        #     marketFactors), suspendTickers].dot(prevPortfolio.loc[suspendTickers, 'USD']/notional).abs().max()))
        # factorCurrencyLimit = max(factorCurrencyLimit, np.square(FU.loc[set(FU.index).intersection(
        #     currencyFactors), suspendTickers].dot(prevPortfolio.loc[suspendTickers, 'USD']/notional).abs().max()))

        longExcludeTickers = self.getExcludeTickers(asOfDate, country, 'LONG')
        longExcludeTickers = sorted(list(set(longExcludeTickers).intersection(tickers)))
        longExcludeTickersIndex = [index for index, x in enumerate(tickers) if x in longExcludeTickers]

        validTickers = self.tm.getValidTickers(asOfDate, country, tickerType=self.tickerType, useCache=self.useCache)
        delistTickers = list(set(suspendTickers).difference(validTickers).union(invalidBarraTickers))
        prevPortfolio.loc[delistTickers, 'USD'] = 0

        minMktCapFloatUSDIn = 3.2e8
        minMktCapFloatUSDOut = 3e8

        minADTVUSDIn = 3.2e5
        minADTVUSDOut = 2.8e5

        existingAlphaTickers = set(newTickers).intersection(prevPortfolio.loc[prevPortfolio['USD'] != 0, :].index)
        newAlphaTickers = set(newTickers).intersection(prevPortfolio.loc[prevPortfolio['USD'] == 0, :].index)
        filteredTickers = set()
        prevDate = self.cc.getPrevDateBefore(asOfDate, country)
        if self.useCache:
            marketCapFloat = self.dl.loadMarketCapFloat(prevDate, country, newTickers)
        else:
            marketCapFloat = self.dataFeeders[(country, 'MKT_CAP_FLOAT_ONLY')].getPreviousDateData(asOfDate, newTickers)
        fx = self.cdp.getSpot(prevDate, country, type='close', currencyType=None)
        marketCapFloatUSD = marketCapFloat.multiply(fx)
        tickersIn = marketCapFloatUSD.index[marketCapFloatUSD < minMktCapFloatUSDIn]
        tickersOut = marketCapFloatUSD.index[marketCapFloatUSD < minMktCapFloatUSDOut]
        filteredTickers = filteredTickers.union(newAlphaTickers.intersection(tickersIn)).union(
            existingAlphaTickers.intersection(tickersOut))

        if self.useCache:
            closePrice = self.dl.loadData('CLOSE', prevDate, prevDate, country, newTickers)
        else:
            closePrice = self.dataFeeders[(country, 'CLOSE')].getPreviousDateData(asOfDate, newTickers)
        fx = self.cdp.getSpot(prevDate, country, type='close', currencyType=None)
        closePriceUSD = closePrice.multiply(fx)
        tickersIn = closePriceUSD.index[closePriceUSD < self.minUnitPriceUSDIn]
        tickersOut = closePriceUSD.index[closePriceUSD < self.minUnitPriceUSDOut]
        filteredTickers = filteredTickers.union(newAlphaTickers.intersection(tickersIn)).union(
            existingAlphaTickers.intersection(tickersOut))

        ADTVUSD = self.getPreviousMeanVolumeLC(asOfDate, country, newTickers, volumeWin=self.adtvFilterWin,
                                volumeWinExtended=None).multiply(fx)
        tickersIn = ADTVUSD.index[ADTVUSD < minADTVUSDIn]
        tickersOut = ADTVUSD.index[ADTVUSD < minADTVUSDOut]
        filteredTickers = filteredTickers.union(newAlphaTickers.intersection(tickersIn)).union(
            existingAlphaTickers.intersection(tickersOut))

        ADTVUSD = ADTVUSD.reindex(newTickers).fillna(np.inf)
        adtvLimitMapping = 5
        adtvLimit = adtvLimitMapping * ADTVUSD * volumeLimit

        tcConstraintLimit = pd.concat([pd.Series(np.inf, index=tickers).to_frame('UPPER'),
                            pd.Series(-np.inf, index=tickers).to_frame('LOWER'), prevPortfolio[['USD']]], axis=1)
        for side in ['BO', 'BC', 'SO', 'SC']:
            restrictedTickers = self.getRestrictedList(asOfDate, country, side)
            restrictedTickers = list(set(restrictedTickers).intersection(tickers))
            if side == 'BO':
                tcConstraintLimit.loc[restrictedTickers, 'UPPER'] = tcConstraintLimit.apply(lambda row: min(max(-row['USD'], 0.0), row['UPPER']), axis=1)
            elif side == 'BC':
                tcConstraintLimit.loc[restrictedTickers, 'UPPER'] = tcConstraintLimit.apply(lambda row: min(0.0 if row['USD'] < 0 else np.inf, row['UPPER']), axis=1)
            elif side == 'SO':
                tcConstraintLimit.loc[restrictedTickers, 'LOWER'] = tcConstraintLimit.apply(lambda row: max(min(-row['USD'], 0.0), row['LOWER']), axis=1)
            elif side == 'SC':
                tcConstraintLimit.loc[restrictedTickers, 'LOWER'] = tcConstraintLimit.apply(lambda row: max(0.0 if row['USD'] > 0 else -np.inf, row['LOWER']), axis=1)
        tcConstraintLimit['UPPER'] = pd.concat([tcConstraintLimit['UPPER'], shortSellData.sub(prevPortfolio)], axis=1).max(axis=1)
        # filteredData = prevPortfolio['USD'].iloc[filteredTickersIndex].abs().reindex(tickers).fillna(0)
        # tcConstraintLimit['UPPER'] = pd.concat([tcConstraintLimit['UPPER'], filteredData.to_frame('USD')], axis=1).max(axis=1)
        # tcConstraintLimit['LOWER'] = pd.concat([tcConstraintLimit['LOWER'], -filteredData.to_frame('USD')], axis=1).min(axis=1)
        posUpper = prevPortfolio['USD'].add(tcConstraintLimit['UPPER'])
        tcConstraintTickersUpper = posUpper[posUpper < 0].index
        posLower = prevPortfolio['USD'].add(tcConstraintLimit['LOWER'])
        tcConstraintTickersLower = posLower[posLower > 0].index
        filteredTickers = filteredTickers.difference(tcConstraintTickersUpper).difference(tcConstraintTickersLower)
        filteredTickersIndex = [index for index, x in enumerate(tickers) if x in filteredTickers]

        if self.useCache:
            roundlot = self.dl.loadRoundlotCache(asOfDate, country)
        else:
            roundlot = self.dl.getRoundLot(asOfDate, country, tickerType='id')
        roundlot = roundlot[tickers]
        if pd.notnull(roundlot).sum() == 0:
            raise Exception('Round lot all missing in %s' % country)
        roundlot = roundlot.fillna(self.defaultRoundLotMapping[country])
        shortSellFileDate = self.getShortSellFileDate(asOfDate, country)
        spot = self.cdp.getSpot(self.cc.getPrevDateBefore(shortSellFileDate, country), country, type='close')
        priceLC = self.getClosePrice(shortSellFileDate, country, tickers)
        roundlotUSD = roundlot.multiply(priceLC).multiply(spot)
        roundlotUSD = roundlotUSD[tickers]

        data_client = MosekApiClient()
        data = [n, nSuspend, asOfDate, alphas, alphaWeight, cost, tcWeight, robustWeight, shortSellData,
            shortSellMultiplier, netNotionalBound, prevPortfolio, notional, tcConstraintLimit, betas, betaExposureLimit,
            FU, betaVarianceLimit, factorsWithoutAlphaViewStyle, factorStyleLimit, factorsWithoutAlphaViewIndustry,
            factorIndustryLimit, longExcludeTickersIndex, singleLimit, tickers, longExcludeTickers, self.firstPositionDate,
            self.tradingLimitUSD, exposure, factorExposureLimit, overallStyleLimit, overallSectorLimit, volumeLimit,
            filteredTickersIndex, factorMarketLimit, factorCountryLimit, factorCurrencyLimit, overallMarketLimit,
            overallCountryLimit, overallCurrencyLimit, marketFactors, countryFactors, currencyFactors, adtvLimit,
            prevAdjFactor, apacMultiplier, countryInflationFactor, srisk, volTarget, scalingCap, riskTargetSmoothing]
        if 'GLOBAL_CONFIG_ENV' in os.environ and os.environ['GLOBAL_CONFIG_ENV'].lower() in {
                'prod', 'staging', 'prod_db_parallel'} or not self.multiProcessMode:
            # print(dt.datetime.now(), 'local mosek')
            portfolio, totalAdjFactor = self.mosekRun(data)
            # print(dt.datetime.now(), portfolio)
        else:
            # print(dt.datetime.now(), 'api mosek')
            portfolio, totalAdjFactor = data_client.mosek_run(data)
            # print(dt.datetime.now(), portfolio)

        combo = pd.DataFrame()
        combo['PORTFOLIO'] = portfolio.iloc[nSuspend:]
        combo['PREV_PORTFOLIO'] = prevPortfolio['USD'].iloc[nSuspend:]
        combo['SHORTSELL_ROUNDLOT'] = shortSellData['USD'].iloc[nSuspend:].div(roundlotUSD.iloc[nSuspend:]).apply(
            lambda x: (math.ceil(x) if not self.adjustByShortSellResponse else round(x)) if pd.notnull(x) else x)
        combo['UPPER'] = tcConstraintLimit['UPPER'].iloc[nSuspend:]
        combo['LOWER'] = tcConstraintLimit['LOWER'].iloc[nSuspend:]
        combo['ROUNDLOT_USD'] = roundlotUSD.iloc[nSuspend:]
        validRoundLot = combo.loc[pd.notnull(combo['ROUNDLOT_USD']), :].index
        combo['TRADE'] = combo['PORTFOLIO'].sub(combo['PREV_PORTFOLIO'])
        combo['TRADE_ROUNDLOT'] = combo['TRADE'].div(combo['ROUNDLOT_USD']).apply(
            lambda x: int(x) if pd.notnull(x) else x)
        combo['PREV_ROUNDLOT'] = combo['PREV_PORTFOLIO'].div(combo['ROUNDLOT_USD']).apply(
            lambda x: round(x) if pd.notnull(x) else x)
        combo['SHORTSELL_BOUND'] = combo['SHORTSELL_ROUNDLOT'].sub(combo['PREV_ROUNDLOT'])
        combo['TRADE_ROUNDLOT'] = combo[['TRADE_ROUNDLOT', 'SHORTSELL_BOUND']].max(axis=1)
        combo['TRADE_ROUNDLOT_PCT'] = combo['TRADE'].div(combo['ROUNDLOT_USD']).sub(combo['TRADE_ROUNDLOT'])
        combo = combo.loc[validRoundLot, :]
        deficit = combo['PORTFOLIO'].sum() - combo['PREV_ROUNDLOT'].add(combo['TRADE_ROUNDLOT']).multiply(
            combo['ROUNDLOT_USD']).sum()
        combo['TRADE_ROUNDLOT_NEW'] = combo['TRADE_ROUNDLOT']
        if deficit > 0:
            combo = combo.sort_values('TRADE_ROUNDLOT_PCT', ascending=False)
            for index, row in combo.iterrows():
                if row['ROUNDLOT_USD'] <= deficit and (row['TRADE_ROUNDLOT_NEW'] + 1) * row[
                        'ROUNDLOT_USD'] <= row['UPPER']:
                    combo.loc[index, 'TRADE_ROUNDLOT_NEW'] += 1
                    deficit -= row['ROUNDLOT_USD']
        else:
            combo = combo.sort_values('TRADE_ROUNDLOT_PCT', ascending=True)
            for index, row in combo.iterrows():
                if row['ROUNDLOT_USD'] <= -deficit and row['PREV_ROUNDLOT'] + row['TRADE_ROUNDLOT_NEW'] > row[
                        'SHORTSELL_ROUNDLOT'] and (row['TRADE_ROUNDLOT_NEW'] - 1) * row[
                        'ROUNDLOT_USD'] >= row['LOWER']:
                    combo.loc[index, 'TRADE_ROUNDLOT_NEW'] -= 1
                    deficit += row['ROUNDLOT_USD']
        combo = combo.loc[portfolio.iloc[nSuspend:].index, :]
        combo['PORTFOLIO_ROUNDLOT'] = combo['PREV_ROUNDLOT'].add(combo['TRADE_ROUNDLOT_NEW'])
        combo['PORTFOLIO_ROUNDLOT'] = combo[['PORTFOLIO_ROUNDLOT', 'SHORTSELL_ROUNDLOT']].max(axis=1)
        combo['PORTFOLIO_ROUNDLOT_USD'] = combo['PORTFOLIO_ROUNDLOT'].multiply(combo['ROUNDLOT_USD'])
        portfolio[validRoundLot] = combo.loc[validRoundLot, 'PORTFOLIO_ROUNDLOT_USD']

        thresholdTinyPosition = 5e-6 if country in ['AUS', 'KOR'] else 1e-6
        portfolio[portfolio.abs() < thresholdTinyPosition * notional] = 0
        portfolioRate = pd.concat([prevPortfolio.loc[prevPortfolioSuspendTickers, 'RATE'], tickerRates[newTickers]], axis=0)
        unitPortfolio = pd.concat([portfolio.to_frame('USD'), portfolioRate.to_frame('RATE')], axis=1)

        factorRisk = np.sqrt(unitPortfolio[['USD']].T.dot(exposureFull.loc[unitPortfolio.index, :].fillna(0)).dot(covFull).dot(
            exposureFull.loc[unitPortfolio.index, :].fillna(0).T).dot(unitPortfolio[['USD']])).iloc[0,0] / notional
        idioRisk = np.sqrt(np.square(srisk.loc[unitPortfolio.index]).multiply(np.square(unitPortfolio['USD'])).sum()) / notional
        totalRisk = np.sqrt(np.square(factorRisk) + np.square(idioRisk))
        threshold = 0.001 * notional
        longNames = (unitPortfolio['USD'] > threshold).sum()
        shortNames = (unitPortfolio['USD'] < -threshold).sum()
        print(dt.datetime.now(), '%s %s %s: Solved factor risk %.3f, Solved idio risk %.3f, Total risk %.3f, Scaling factor %.3f' %
              (country, asOfDate.strftime('%Y-%m-%d'), backtestVersion, factorRisk, idioRisk, totalRisk, totalAdjFactor))
        target = self.loadTarget(country, backtestVersion)
        nextDay = self.cc.getNextDateAfter(asOfDate, country)
        target.loc[nextDay, 'Solved Notional'] = combo['PORTFOLIO'].abs().sum() / 2
        target.loc[nextDay, 'Solved Factor Risk'] = factorRisk
        target.loc[nextDay, 'Solved Idio Risk'] = idioRisk
        target.loc[nextDay, 'Solved Total Risk'] = totalRisk
        target.loc[nextDay, 'Final Notional'] = unitPortfolio['USD'].abs().sum() / 2
        target.loc[nextDay, 'Number of Long'] = longNames
        target.loc[nextDay, 'Number of Short'] = shortNames
        target.loc[nextDay, 'Total Adj Factor'] = totalAdjFactor
        target.loc[nextDay, 'APAC Multiplier'] = apacMultiplier
        self.saveTarget(target, country, backtestVersion)
        return unitPortfolio

    def mosekRun(self, data):
        [n, nSuspend, asOfDate, alphas, alphaWeight, cost, tcWeight, robustWeight, shortSellData,
         shortSellMultiplier, netNotionalBound, prevPortfolio, notional, tcConstraintLimit, betas, betaExposureLimit,
         FU, betaVarianceLimit, factorsWithoutAlphaViewStyle, factorStyleLimit, factorsWithoutAlphaViewIndustry,
         factorIndustryLimit, longExcludeTickersIndex, singleLimit, tickers, longExcludeTickers, firstUnitNotionalDate,
         tradingLimitUSD, exposure, factorExposureLimit, overallStyleLimit, overallSectorLimit, volumeLimit,
         filteredTickersIndex, factorMarketLimit, factorCountryLimit, factorCurrencyLimit, overallMarketLimit,
         overallCountryLimit, overallCurrencyLimit, marketFactors, countryFactors, currencyFactors, adtvLimit,
         prevAdjFactor, apacMultiplier, countryInflationFactor, srisk, volTarget, scalingCap, riskTargetSmoothing,
         ] = data

        M = Model()
        # M.setLogHandler(sys.stdout)
        M.setSolverParam('intpntSolveForm', 'primal')
        M.setSolverParam('licenseWait', 'on')
        M.setSolverParam('cacheLicense', 'off')

        # Defines the variables (holdings).
        x = M.variable('x', n, Domain.unbounded())
        tc = M.variable('tc', n - nSuspend, Domain.unbounded())
        re = M.variable('re', 1, Domain.unbounded())
        ss = M.variable('ss', n - nSuspend, Domain.lessThan(0.0))
        # p = M.variable('p', 1, Domain.unbounded())
        # q = M.variable('q', 1, Domain.unbounded())
        # r = M.variable('r', 1, Domain.unbounded())

        # Notional constraint
        if asOfDate > firstUnitNotionalDate:
            M.constraint('net upper', Expr.sum(x), Domain.lessThan(netNotionalBound))
            M.constraint('net lower', Expr.sum(x), Domain.greaterThan(-netNotionalBound))
        else:
            M.constraint('neutral', Expr.sum(x), Domain.equalsTo(0.0))

        # Suspend position remain unchanged
        M.constraint('suspend', x.slice(0, nSuspend),
                     Domain.equalsTo(prevPortfolio.iloc[:nSuspend, 0].div(notional).values.tolist()))

        # Short sell constraint
        M.constraint('short sell', x, Domain.greaterThan(shortSellData['USD'].div(notional).values.tolist()))

        # Short sell cost
        M.constraint('short sell bound', Expr.sub(ss, x.slice(nSuspend, n)), Domain.lessThan(0.0))

        if asOfDate > firstUnitNotionalDate:
            # Transaction cost
            M.constraint('tc upper', Expr.sub(tc, Expr.sub(x.slice(nSuspend, n), prevPortfolio.iloc[nSuspend:n, 0].div(
                notional).values.tolist())), Domain.greaterThan(0.0))
            M.constraint('tc lower', Expr.sub(tc, Expr.sub(prevPortfolio.iloc[nSuspend:n, 0].div(notional).values.tolist(),
                                                           x.slice(nSuspend, n))), Domain.greaterThan(0.0))

            # Restricted list
            M.constraint('restricted upper', Expr.sub(x.slice(nSuspend, n),
                                                      prevPortfolio['USD'].add(tcConstraintLimit['UPPER']).iloc[
                                                      nSuspend:n].div(notional).values.tolist()), Domain.lessThan(0.0))
            M.constraint('restricted lower', Expr.sub(x.slice(nSuspend, n),
                                                      prevPortfolio['USD'].add(tcConstraintLimit['LOWER']).iloc[
                                                      nSuspend:n].div(notional).values.tolist()), Domain.greaterThan(0.0))

        # Robust estimation
        M.constraint('robust', Expr.vstack(0.5, re, x.slice(nSuspend, n)), Domain.inRotatedQCone())

        # Single limit
        # M.constraint('single limit upper', x.slice(nSuspend, n), Domain.lessThan(2.0 * notionalMultiplier[country] * self.singleLimit))
        # M.constraint('single limit lower', x.slice(nSuspend, n), Domain.greaterThan(-2.0 * notionalMultiplier[country] * self.singleLimit))

        # Beta exposure
        M.constraint('beta upper', Expr.dot(x, betas.values.tolist()), Domain.lessThan(betaExposureLimit))
        M.constraint('beta lower', Expr.dot(x, betas.values.tolist()), Domain.greaterThan(-betaExposureLimit))

        # Imposes a bound on factor risk
        # M.constraint('factor risk', Expr.vstack(0.5, p, Expr.mul(FU.values.tolist(), x)), Domain.inRotatedQCone())
        # M.constraint('industry factor risk', Expr.vstack(0.5, industryVarianceLimit, Expr.mul(FU.loc[set(validFactors[validFactors].index).intersection(factorsWithoutAlphaViewIndustry), :].values.tolist(), x.slice(nSuspend, n))), Domain.inRotatedQCone())

        # Imposes a bound on beta risk
        M.constraint('beta risk upper', Expr.dot(FU.loc['BETA', :].values.tolist(), x),
                     Domain.lessThan(np.sqrt(betaVarianceLimit)))
        M.constraint('beta risk lower', Expr.dot(FU.loc['BETA', :].values.tolist(), x),
                     Domain.greaterThan(-np.sqrt(betaVarianceLimit)))

        # Imposes a bound on standalone exposure
        M.constraint('style exposure upper', Expr.mul(exposure.T.values.tolist(), x), Domain.lessThan(
            np.sqrt(factorExposureLimit)))
        M.constraint('style exposure lower', Expr.mul(exposure.T.values.tolist(), x), Domain.greaterThan(
            -np.sqrt(factorExposureLimit)))

        if overallStyleLimit < np.inf:
            M.constraint('overall style risk', Expr.vstack(0.5, overallStyleLimit, Expr.mul(
                FU.loc[set(FU.index).intersection(factorsWithoutAlphaViewStyle), :].values.tolist(), x)),
                Domain.inRotatedQCone())

        if overallSectorLimit < np.inf:
            M.constraint('overall Sector risk', Expr.vstack(0.5, overallSectorLimit, Expr.mul(
                FU.loc[set(FU.index).intersection(factorsWithoutAlphaViewIndustry), :].values.tolist(), x)),
                Domain.inRotatedQCone())

        if overallMarketLimit < np.inf:
            M.constraint('overall Market risk', Expr.vstack(0.5, overallMarketLimit, Expr.mul(
                FU.loc[set(FU.index).intersection(marketFactors), :].values.tolist(), x)),
                Domain.inRotatedQCone())

        if overallCountryLimit < np.inf:
            M.constraint('overall Country risk', Expr.vstack(0.5, overallCountryLimit, Expr.mul(
                FU.loc[set(FU.index).intersection(countryFactors), :].values.tolist(), x)),
                Domain.inRotatedQCone())

        if overallCurrencyLimit < np.inf:
            M.constraint('overall Currency risk', Expr.vstack(0.5, overallCurrencyLimit, Expr.mul(
                FU.loc[set(FU.index).intersection(currencyFactors), :].values.tolist(), x)),
                Domain.inRotatedQCone())

        # Imposes a bound on standalone style risk
        intersectFactors = list(set(FU.index).intersection(factorsWithoutAlphaViewStyle))
        M.constraint('style risk upper', Expr.sub(Expr.mul(FU.loc[intersectFactors, :].values.tolist(), x),
            factorStyleLimit[intersectFactors].values.tolist()), Domain.lessThan(0.0))
        M.constraint('style risk lower', Expr.add(Expr.mul(FU.loc[intersectFactors, :].values.tolist(), x),
            factorStyleLimit[intersectFactors].values.tolist()), Domain.greaterThan(0.0))

        # Impose long excluded names constraint
        M.constraint('long exclude names', x.pick(longExcludeTickersIndex), Domain.lessThan(0.0))

        # Impose filtered tickers constraint
        M.constraint('filtered tickers', x.pick(filteredTickersIndex), Domain.equalsTo(0.0))

        # Impose ADTV limit constraint
        M.constraint('ADTV limit upper', Expr.add(x.slice(nSuspend, n), (-adtvLimit).values.tolist()), Domain.lessThan(0.0))
        M.constraint('ADTV limit lower', Expr.add(x.slice(nSuspend, n), adtvLimit.values.tolist()), Domain.greaterThan(0.0))

        # Imposes a bound on idiosyncratic risk
        # M.constraint('idio risk', Expr.vstack(0.5, q, Expr.mulElm(srisk.values.tolist(), x)), Domain.inRotatedQCone())

        # Impose total variance bound
        # M.constraint('total risk', Expr.sub(Expr.add(p, q), r), Domain.lessThan(0.0))

        # Single limit
        singleLimitUpper = pd.Series()
        singleLimitLower = pd.Series()
        singleLimitUpper = singleLimitUpper.add(prevPortfolio['USD'].add(
            tcConstraintLimit['LOWER'], fill_value=0).div(notional), fill_value=0)
        singleLimitLower = singleLimitLower.add(prevPortfolio['USD'].add(
            tcConstraintLimit['UPPER'], fill_value=0).div(notional), fill_value=0)
        singleLimitUpper = singleLimitUpper.apply(lambda x: max(x, singleLimit))
        singleLimitLower = singleLimitLower.apply(lambda x: min(x, -singleLimit))
        M.constraint('single limit upper', Expr.sub(x.slice(nSuspend, n),
                                                    singleLimitUpper.iloc[nSuspend:].values.tolist()), Domain.lessThan(0.0))
        M.constraint('single limit lower', Expr.sub(x.slice(nSuspend, n),
                                                    singleLimitLower.iloc[nSuspend:].values.tolist()),
                     Domain.greaterThan(0.0))

        M1 = M.clone()

        # Impose bound on standalone sector risk
        oldfactorIndustryLimit = factorIndustryLimit
        M.constraint('sector risk upper', Expr.mul(FU.loc[set(FU.index).intersection(
            factorsWithoutAlphaViewIndustry), :].values.tolist(), x), Domain.lessThan(np.sqrt(factorIndustryLimit)))
        M.constraint('sector risk lower', Expr.mul(FU.loc[set(FU.index).intersection(
            factorsWithoutAlphaViewIndustry), :].values.tolist(), x), Domain.greaterThan(-np.sqrt(factorIndustryLimit)))

        # Imposes a bound on standalone market risk
        oldfactorMarketLimit = factorMarketLimit
        M.constraint('market risk upper', Expr.mul(FU.loc[set(FU.index).intersection(
            marketFactors), :].values.tolist(), x), Domain.lessThan(
            np.sqrt(factorMarketLimit)))
        M.constraint('market risk lower', Expr.mul(FU.loc[set(FU.index).intersection(
            marketFactors), :].values.tolist(), x), Domain.greaterThan(
            -np.sqrt(factorMarketLimit)))

        # Imposes a bound on standalone country risk
        oldfactorCountryLimit = factorCountryLimit
        M.constraint('country risk upper', Expr.mul(FU.loc[set(FU.index).intersection(
            countryFactors), :].values.tolist(), x), Domain.lessThan(
            np.sqrt(factorCountryLimit)))
        M.constraint('country risk lower', Expr.mul(FU.loc[set(FU.index).intersection(
            countryFactors), :].values.tolist(), x), Domain.greaterThan(
            -np.sqrt(factorCountryLimit)))

        # Imposes a bound on standalone currency risk
        oldfactorCurrencyLimit = factorCurrencyLimit
        M.constraint('currency risk upper', Expr.mul(FU.loc[set(FU.index).intersection(
            currencyFactors), :].values.tolist(), x), Domain.lessThan(
            np.sqrt(factorCurrencyLimit)))
        M.constraint('currency risk lower', Expr.mul(FU.loc[set(FU.index).intersection(
            currencyFactors), :].values.tolist(), x), Domain.greaterThan(
            -np.sqrt(factorCurrencyLimit)))

        # Volume limit
        tradingLimitUSD = tradingLimitUSD * volumeLimit
        tradingLimitUSD = tradingLimitUSD.reindex(tickers).fillna(np.inf)
        oldTradingLimit = deepcopy(tradingLimitUSD)
        diffShortSellData = shortSellData['USD'].sub(prevPortfolio['USD']).apply(lambda x: max(x, 0))
        diffSingleLimit = prevPortfolio['USD'].abs()
        longExcludeData = prevPortfolio['USD'].loc[longExcludeTickers].reindex(tickers).fillna(0)
        filteredData = prevPortfolio['USD'].iloc[filteredTickersIndex].abs().reindex(tickers).fillna(0)
        diffADTVData = prevPortfolio['USD'].abs()
        diffADTVData.iloc[nSuspend:] = diffADTVData.iloc[nSuspend:] - adtvLimit.abs()
        tradingLimitUSD = pd.concat([tradingLimitUSD, diffShortSellData.to_frame('USD')], axis=1).max(axis=1)
        tradingLimitUSD = pd.concat([tradingLimitUSD, diffSingleLimit.sub(singleLimit * notional).to_frame('USD')],
                                    axis=1).max(axis=1)
        tradingLimitUSD = pd.concat([tradingLimitUSD, longExcludeData.to_frame('USD')], axis=1).max(axis=1)
        tradingLimitUSD = pd.concat([tradingLimitUSD, filteredData.to_frame('USD')], axis=1).max(axis=1)
        tradingLimitUSD = pd.concat([tradingLimitUSD, diffADTVData.to_frame('USD')], axis=1).max(axis=1)
        M.constraint('volume upper', Expr.sub(tc, tradingLimitUSD.div(notional).iloc[nSuspend:].values.tolist()),
                     Domain.lessThan(0.0))

        M2 = M.clone()

        totalAdjFactor = prevAdjFactor

        alphaWeightOld = alphaWeight
        shortSellMultiplierOld = shortSellMultiplier
        robustWeightOld = robustWeight
        tcWeightOld = tcWeight

        alphaWeight = alphaWeightOld / totalAdjFactor
        shortSellMultiplier = shortSellMultiplierOld / totalAdjFactor
        robustWeight = robustWeightOld / totalAdjFactor
        tcWeight = tcWeightOld / totalAdjFactor

        # objective
        if asOfDate > firstUnitNotionalDate:
            M.objective('obj', ObjectiveSense.Maximize, Expr.add([
                Expr.dot(alphas.mul(alphaWeight).iloc[nSuspend:].values.tolist(), x.slice(nSuspend, n)),
                Expr.mul(-cost * tcWeight, Expr.sum(tc)),
                Expr.mul(-robustWeight * alphaWeight, re),
                Expr.mul(shortSellData['RATE'].mul(shortSellMultiplier).iloc[nSuspend:].values.tolist(), ss),
            ]))
        else:
            M.objective('obj', ObjectiveSense.Maximize, Expr.add([
                Expr.dot(alphas.mul(alphaWeight).iloc[nSuspend:].values.tolist(), x.slice(nSuspend, n)),
                Expr.mul(-robustWeight * alphaWeight, re),
                Expr.mul(shortSellData['RATE'].mul(shortSellMultiplier).iloc[nSuspend:].values.tolist(), ss),
            ]))

        # M.writeTask(self.staticConfig.find(self.broker.lower()).find('analysis').contents[0] + '/dump.task.gz')
        M.solve()

        x = M.getVariable('x')
        portfolio = pd.Series(x.level(), index=tickers)
        factorRisk = np.square(FU.dot(portfolio.to_frame())).sum().iloc[0]
        idioRisk = np.square(srisk.loc[portfolio.index]).multiply(np.square(portfolio)).sum()
        totalRisk = np.sqrt(factorRisk + idioRisk)
        oldTotalAdjFactor = totalAdjFactor
        totalAdjFactor = oldTotalAdjFactor * volTarget * apacMultiplier / countryInflationFactor / totalRisk
        totalAdjFactor = riskTargetSmoothing * totalAdjFactor + (1 - riskTargetSmoothing) * oldTotalAdjFactor
        totalAdjFactor = min(totalAdjFactor, scalingCap)
        # print(dt.datetime.now(), '%s : Presolved total risk %.3f, Old scaling factor %.3f, New scaling factor %.3f' %
        #       (asOfDate.strftime('%Y-%m-%d'), totalRisk, oldTotalAdjFactor, totalAdjFactor))

        M.dispose()
        M = M2.clone()
        alphaWeight = alphaWeightOld / totalAdjFactor
        shortSellMultiplier = shortSellMultiplierOld / totalAdjFactor
        robustWeight = robustWeightOld / totalAdjFactor
        tcWeight = tcWeightOld / totalAdjFactor

        # objective
        if asOfDate > firstUnitNotionalDate:
            M.objective('obj', ObjectiveSense.Maximize, Expr.add([
                Expr.dot(alphas.mul(alphaWeight).iloc[nSuspend:].values.tolist(), M.getVariable('x').slice(nSuspend, n)),
                Expr.mul(-cost * tcWeight, Expr.sum(M.getVariable('tc'))),
                Expr.mul(-robustWeight * alphaWeight, M.getVariable('re')),
                Expr.mul(shortSellData['RATE'].mul(shortSellMultiplier).iloc[nSuspend:].values.tolist(), M.getVariable('ss')),
            ]))
        else:
            M.objective('obj', ObjectiveSense.Maximize, Expr.add([
                Expr.dot(alphas.mul(alphaWeight).iloc[nSuspend:].values.tolist(), M.getVariable('x').slice(nSuspend, n)),
                Expr.mul(-robustWeight * alphaWeight, M.getVariable('re')),
                Expr.mul(shortSellData['RATE'].mul(shortSellMultiplier).iloc[nSuspend:].values.tolist(), M.getVariable('ss')),
            ]))

        # M.writeTask(self.staticConfig.find(self.broker.lower()).find('analysis').contents[0] + '/dump.task.gz')
        M.solve()

        nTry = 0
        maxTry = 20
        while M.getPrimalSolutionStatus() not in [SolutionStatus.Optimal, SolutionStatus.NearOptimal] and nTry <= maxTry:
            print(dt.datetime.now(), 'Solution cannot be found, relaxing sector and volume limit, number of try:', nTry)
            M.dispose()
            M = M1.clone()
            factorIndustryLimit += oldfactorIndustryLimit
            factorMarketLimit += oldfactorMarketLimit
            factorCountryLimit += oldfactorCountryLimit
            factorCurrencyLimit += oldfactorCurrencyLimit
            tradingLimitUSD += oldTradingLimit
            M.constraint('sector risk upper', Expr.mul(FU.loc[set(FU.index).intersection(
                factorsWithoutAlphaViewIndustry), :].values.tolist(), M.getVariable('x')), Domain.lessThan(
                np.sqrt(factorIndustryLimit)))
            M.constraint('sector risk lower', Expr.mul(FU.loc[set(FU.index).intersection(
                factorsWithoutAlphaViewIndustry), :].values.tolist(), M.getVariable('x')), Domain.greaterThan(
                -np.sqrt(factorIndustryLimit)))
            M.constraint('market risk upper', Expr.mul(FU.loc[set(FU.index).intersection(
                marketFactors), :].values.tolist(), M.getVariable('x')), Domain.lessThan(
                np.sqrt(factorMarketLimit)))
            M.constraint('market risk lower', Expr.mul(FU.loc[set(FU.index).intersection(
                marketFactors), :].values.tolist(), M.getVariable('x')), Domain.greaterThan(
                -np.sqrt(factorMarketLimit)))
            M.constraint('country risk upper', Expr.mul(FU.loc[set(FU.index).intersection(
                countryFactors), :].values.tolist(), M.getVariable('x')), Domain.lessThan(
                np.sqrt(factorCountryLimit)))
            M.constraint('country risk lower', Expr.mul(FU.loc[set(FU.index).intersection(
                countryFactors), :].values.tolist(), M.getVariable('x')), Domain.greaterThan(
                -np.sqrt(factorCountryLimit)))
            M.constraint('currency risk upper', Expr.mul(FU.loc[set(FU.index).intersection(
                currencyFactors), :].values.tolist(), M.getVariable('x')), Domain.lessThan(
                np.sqrt(factorCurrencyLimit)))
            M.constraint('currency risk lower', Expr.mul(FU.loc[set(FU.index).intersection(
                currencyFactors), :].values.tolist(), M.getVariable('x')), Domain.greaterThan(
                -np.sqrt(factorCurrencyLimit)))
            M.constraint('volume upper', Expr.sub(M.getVariable('tc'),
                tradingLimitUSD.div(notional).iloc[nSuspend:].values.tolist()), Domain.lessThan(0.0))
            # objective
            if asOfDate > firstUnitNotionalDate:
                M.objective('obj', ObjectiveSense.Maximize, Expr.add([
                    Expr.dot(alphas.mul(alphaWeight).iloc[nSuspend:].values.tolist(), M.getVariable('x').slice(nSuspend, n)),
                    Expr.mul(-cost * tcWeight, Expr.sum(M.getVariable('tc'))),
                    Expr.mul(-robustWeight * alphaWeight, M.getVariable('re')),
                    Expr.mul(shortSellData['RATE'].mul(shortSellMultiplier).iloc[nSuspend:].values.tolist(), M.getVariable('ss')),
                ]))
            else:
                M.objective('obj', ObjectiveSense.Maximize, Expr.add([
                    Expr.dot(alphas.mul(alphaWeight).iloc[nSuspend:].values.tolist(), M.getVariable('x').slice(nSuspend, n)),
                    Expr.mul(-robustWeight * alphaWeight, M.getVariable('re')),
                    Expr.mul(shortSellData['RATE'].mul(shortSellMultiplier).iloc[nSuspend:].values.tolist(), M.getVariable('ss')),
                ]))
            M.solve()
            nTry += 1

        x = M.getVariable('x')
        portfolio = pd.Series(x.level() * notional, index=tickers)
        M.dispose()
        M1.dispose()
        M2.dispose()
        return portfolio, totalAdjFactor


    def constructUnitPortfolio(self, asOfDate, country, tickerRates, positionType, backtestVersion=''):
        prevPortfolio = self.getReevaluatedPrevPortfolio(asOfDate, country, positionType, backtestVersion=backtestVersion)
        prevPortfolioSuspend, _ = self.separatePortfolioBySuspend(asOfDate, country, prevPortfolio)
        prevPortfolioSuspendTickers = prevPortfolioSuspend.index
        if country == 'KOR' and asOfDate >= dt.datetime(2011, 8, 9) and asOfDate <= dt.datetime(2011, 11, 8) or asOfDate == dt.datetime(2018, 12, 31):
            print(dt.datetime.now(), '%s Skipped %s' % (country, asOfDate.strftime('%Y-%m-%d')))
            unitPortfolio = prevPortfolio
        elif len(prevPortfolioSuspendTickers) > 0.5 * tickerRates.shape[0] or \
                self.shortSellData['USD'].sum() < 0.5 * self.nt.getNotional(asOfDate, country, 'model') and \
                not self.adjustByShortSellResponse or \
                tickerRates.shape[0] == 0:
            if self.liveMode:
                print(dt.datetime.now(), 'len(prevPortfolioSuspendTickers)')
                print(dt.datetime.now(), len(prevPortfolioSuspendTickers))
                print(dt.datetime.now(), 'tickerRates.shape[0]')
                print(dt.datetime.now(), tickerRates.shape[0])
                print(dt.datetime.now(), "self.shortSellData['USD'].sum()")
                print(dt.datetime.now(), self.shortSellData['USD'].sum())
                print(dt.datetime.now(), "self.nt.getNotional(asOfDate, country, 'model')")
                print(dt.datetime.now(), self.nt.getNotional(asOfDate, country, 'model'))
                print(dt.datetime.now(), 'tickerRates.shape[0]')
                print(dt.datetime.now(), tickerRates.shape[0])
                raise RuntimeError('%s %s portfolio has too much suspend tickers or short sell inventory is too small' % (country, asOfDate.strftime('%Y-%m-%d')))
            else:
                print(dt.datetime.now(), '%s %s portfolio has too much suspend tickers or short sell inventory is too small' % (country, asOfDate.strftime('%Y-%m-%d')))
                unitPortfolio = prevPortfolio
        else:
            unitPortfolio = self.solvePortfolio(asOfDate, country, prevPortfolio, prevPortfolioSuspendTickers, tickerRates, backtestVersion=backtestVersion)
            unitPortfolio = unitPortfolio.sort_index()
        return unitPortfolio

    def getAlphaTickerRates(self, asOfDate, country):
        tickers = self.getAlphas(asOfDate, country).index
        shortSellData = self.getShortSellQuantityData(asOfDate, country)
        tickerRates = pd.Series(0, index=tickers)
        tickersShort = list(set(tickers).intersection(shortSellData.index))
        tickerRates[tickersShort] = shortSellData.loc[tickersShort, 'RATE']
        return tickerRates

    def getRestrictedList(self, asOfDate, country, side):
        if self.useCache:
            tickersRestictedData = self.dl.loadModelPeriodEventCache(asOfDate, country, self.broker, 'RESTRICTION')
        else:
            tickersRestictedData = self.dl.loadModelPeriodEventData(asOfDate, self.broker, self.tickerType,
                                                                    'RESTRICTION')
        tickersRestictedData['RESTRICTION'] = tickersRestictedData['RESTRICTION'].apply(lambda x: x.split(','))
        tickersRestictedData = tickersRestictedData.loc[tickersRestictedData['RESTRICTION'].apply(lambda x: side in x), :]
        return tickersRestictedData.index.tolist()

    def allocateTickerRates(self, asOfDate, country, suspendSet,):
        alphaTickers = self.getAlphaTickerRates(asOfDate, country)
        tickers = sorted(list(set(alphaTickers.index).difference(suspendSet)))
        shortTickerRates = alphaTickers[tickers]
        return shortTickerRates

    def getClosePrice(self, asOfDate, country, tickers):
        prevDate = self.cc.getPrevDateBefore(asOfDate, country)
        if self.useCache:
            closePrice = self.dl.loadData('CLOSE', prevDate, prevDate, country, tickers)
        else:
            closePrice = self.dl.getFilteredMarketData(prevDate, prevDate, self.tickersByCountry[country], self.tickerType,
                        ['CLOSE']).loc[prevDate].unstack().T.loc[:, 'CLOSE']
        return closePrice

    def getStockSplit(self, asOfDate, country, tickerType):
        if self.useCache:
            caData = self.dl.loadData('STOCK_SPLIT_RATE', asOfDate, asOfDate, country)
        else:
            caData = self.dl.getCAData(asOfDate, asOfDate, country, tickerType)
        if caData.empty:
            stockSplit = pd.Series()
        else:
            if self.useCache:
                stockSplit = caData
            else:
                stockSplit = caData['STOCK_SPLIT_RATE'].loc[asOfDate, :]
            stockSplit = stockSplit[pd.notnull(stockSplit)]
            stockSplit = stockSplit[stockSplit != 1]
        return stockSplit

    def updateShortSellQuantityWithPreinventory(self, asOfDate, country, quantityData):
        prevDate = (asOfDate + pd.tseries.offsets.BDay(-1)).to_pydatetime()
        if prevDate >= self.launchDateDict[country]:
            if country == 'KOR' and asOfDate > dt.datetime(2020, 6, 4) and asOfDate < dt.datetime(2021, 5, 3):
                loadDate = dt.datetime(2020, 6, 4)
            else:
                loadDate = asOfDate
            prevInventory = self.dl.getPrevActualInventory(loadDate, country, self.broker, tickerType=self.tickerType,
                        useCache=self.useCache)
            prevTickersShort = prevInventory.index[prevInventory['QUANTITY'] < 0]
            prevShortSellQuantityData = pd.concat([-prevInventory.loc[prevTickersShort, 'QUANTITY'].to_frame('QUANTITY'), prevInventory.loc[prevTickersShort, 'RATE'].to_frame('RATE')], axis=1)
            if not self.beforeMarketOpen:
                stockSplit = self.getStockSplit(asOfDate, country, self.tickerType)
                splitTickers = list(set(prevShortSellQuantityData.index).intersection(stockSplit.index))
                prevShortSellQuantityData.loc[splitTickers, 'QUANTITY'] = prevShortSellQuantityData.loc[
                    splitTickers, 'QUANTITY'].multiply(stockSplit[splitTickers]).apply(lambda x: int(x))
            tickersPrevCarried = list(set(prevShortSellQuantityData.index).difference(quantityData.index))
            adjQuantityData = pd.concat([quantityData, prevShortSellQuantityData.loc[tickersPrevCarried, :]], axis=0)
            adjQuantity = quantityData['QUANTITY'].append(prevShortSellQuantityData['QUANTITY'])
            adjQuantity = adjQuantity.groupby(adjQuantity.index).sum()
            adjQuantityData.loc[adjQuantity.index, 'QUANTITY'] = adjQuantity
        else:
            adjQuantityData = deepcopy(quantityData)
        return adjQuantityData

    def loadResponseData(self, asOfDate, country):
        data = self.sh.getResponseData(asOfDate, country, tickerType=self.tickerType, useCache=self.useCache)
        return data

    def getResponseLimit(self, asOfDate, country):
        combined = self.sh.getRequestResponseData(asOfDate, country, self.tickerType, useCache=self.useCache).fillna(0)
        diffData = combined['RESPONSE'].sub(combined['REQUEST'])
        limitTickers = diffData.index[diffData<0]
        responseLimit = combined.loc[limitTickers, 'RESPONSE'].sort_index()
        return responseLimit

    def getShortSellFileDate(self, asOfDate, country):
        if self.shortSellInventoryDate is None:
            shortSellFileDate = self.dl.getShortSellFileDate(asOfDate, country, self.broker)
        else:
            shortSellFileDate = self.shortSellInventoryDate
        return shortSellFileDate

    def getShortSellQuantityData(self, asOfDate, country):
        if self.shortSellUnconstraintMode:
            allTickers = self.tickersByCountry[country]
            shortSellQuantityData = pd.Series(np.inf, index=allTickers).to_frame('QUANTITY')
            shortSellQuantityData['RATE'] = self.defaultShortSellRateForUnconstraint
        else:
            if self.adjustByShortSellResponse:
                shortSellFileDate = self.getShortSellFileDate(asOfDate, country)
                responseData = self.loadResponseData(asOfDate, country)
                fileQuantity = responseData['QUANTITY'].groupby(responseData.index).sum()
                fileData = self.dl.loadShortSellInventoryData(shortSellFileDate, country, self.broker, self.tickerType,
                            includePrevinventoryShortSell=False, useCache=self.useCache)
                fileRate = fileData['RATE'].groupby(fileData.index).max()
                fileRate = fileRate.loc[responseData.index]
                shortSellQuantityData = pd.concat([fileQuantity.to_frame('QUANTITY'), fileRate.to_frame('RATE')], axis=1).sort_index()
            else:
                shortSellFileDate = self.getShortSellFileDate(asOfDate, country)
                fileData = self.dl.loadShortSellInventoryData(shortSellFileDate, country, self.broker, self.tickerType,
                            includePrevinventoryShortSell=False, useCache=self.useCache)
                fileQuantity = fileData['QUANTITY'].groupby(fileData.index).sum()
                fileRate = fileData['RATE'].groupby(fileData.index).max()
                shortSellQuantityData = pd.concat([fileQuantity.to_frame('QUANTITY'), fileRate.to_frame('RATE')], axis=1).sort_index()
                if country in self.shortSellRateCapMapping:
                    shortSellQuantityData = shortSellQuantityData.loc[shortSellQuantityData['RATE'] <= self.shortSellRateCapMapping[country], :]
                if self.imposeResponseCut:
                    responseLimit = self.getResponseLimit(asOfDate, country)
                    responseIndex = list(set(shortSellQuantityData.index).intersection(responseLimit.index))
                    if len(responseIndex) > 0:
                        shortSellQuantityData.loc[responseIndex, 'QUANTITY'] = responseLimit[responseIndex]
            if self.includePrevinventoryShortSell:
                shortSellQuantityData = self.updateShortSellQuantityWithPreinventory(asOfDate, country, shortSellQuantityData)
            shortSellQuantityData = shortSellQuantityData.loc[shortSellQuantityData['QUANTITY'] != 0, :]
        return shortSellQuantityData

    def getShortSellUSDData(self, asOfDate, country):
        if self.shortSellUnconstraintMode:
            allTickers = self.tickersByCountry[country]
            shortSellUSDData = pd.Series(np.inf, index=allTickers).to_frame('USD')
            shortSellUSDData['RATE'] = self.defaultShortSellRateForUnconstraint
        else:
            shortSellQuantityData = self.getShortSellQuantityData(asOfDate, country)
            shortSellFileDate = self.getShortSellFileDate(asOfDate, country)
            spot = self.cdp.getSpot(self.cc.getPrevDateBefore(shortSellFileDate, country), country, type='close')
            priceLC = self.getClosePrice(shortSellFileDate, country, shortSellQuantityData.index)
            if self.debugMode:
                print(dt.datetime.now(), 'priceLC', country, shortSellFileDate)
            shortSellUSD = shortSellQuantityData['QUANTITY'].multiply(priceLC).multiply(spot)
            shortSellUSDData = pd.concat([shortSellUSD.to_frame('USD'), shortSellQuantityData['RATE']], axis=1)
            shortSellUSDData = shortSellUSDData.loc[pd.notnull(shortSellUSDData['USD']), :]
        return shortSellUSDData

    def getUnitPortfolio(self, asOfDate, country, positionType, backtestVersion=''):
        suspendSet = self.loadSuspendSet(asOfDate, country)
        tickerRates = self.allocateTickerRates(asOfDate, country, suspendSet)
        portfolio = self.constructUnitPortfolio(asOfDate, country, tickerRates, positionType, backtestVersion=backtestVersion)
        return portfolio

    def save(self, data, asOfDate, country, positionType, backtestVersion=''):
        if backtestVersion is None:
            fileName = '%s/%s/%s/position_%s_%s_%s.csv' % (
                self.positionRoot, country, positionType, positionType, country, asOfDate.strftime('%Y%m%d'))
        else:
            fileName = '%s/%s/%s/%s/position_%s_%s_%s.csv' % (
                self.positionRoot, backtestVersion, country, positionType, positionType, country,
                asOfDate.strftime('%Y%m%d'))
        os.makedirs(os.path.dirname(fileName), exist_ok=True)
        fileWriter.to_csv(data, fileName, header=True, index=True)
        if self.backupMode:
            self.bf.backupFile(fileName, country)

    def getMeanVolumeLCData(self, asOfDate, country, tickers, win):
        if self.useCache:
            volumeData = self.dl.loadData('VOLUME',
                    self.cc.getPrevDateBefore(asOfDate, country, win-1),
                    self.cc.getPrevDateBefore(asOfDate, country), country, tickers)
            closeData = self.dl.loadData('CLOSE',
                    self.cc.getPrevDateBefore(asOfDate, country, win-1),
                    self.cc.getPrevDateBefore(asOfDate, country), country, tickers)
        else:
            volumeData = self.dataFeeders[(country, 'VOLUME')].getDataBefore(asOfDate, tickers, n=win-1)
            closeData = self.dataFeeders[(country, 'CLOSE')].getDataBefore(asOfDate, tickers, n=win-1)
        data = closeData.multiply(volumeData)
        data.replace(to_replace=0, value=np.nan, inplace=True)
        meanVolume = data.mean(axis=0)
        return meanVolume

    def getPreviousMeanVolumeLC(self, asOfDate, country, tickers, volumeWin, volumeWinExtended=None):
        meanVolume = self.getMeanVolumeLCData(asOfDate, country, tickers, volumeWin)
        extendedTickers = meanVolume.index[pd.isnull(meanVolume) | (meanVolume == 0)].tolist()
        if extendedTickers and volumeWinExtended is not None:
            nonExtendedTickers = list(set(meanVolume.index).difference(extendedTickers))
            meanVolumeExtended = self.getMeanVolumeLCData(asOfDate, country, extendedTickers, volumeWinExtended)
            meanVolume = meanVolume[nonExtendedTickers].append(meanVolumeExtended)
            meanVolume.sort_index(inplace=True)
        return meanVolume

    def separatePortfolioBySuspend(self, asOfDate, country, portfolio):
        suspendSet = self.loadSuspendSet(asOfDate, country)
        portfolioSuspend = portfolio.loc[sorted(list(suspendSet.intersection(portfolio.index))), :]
        portfolioNonsuspend = portfolio.loc[sorted(list(set(portfolio.index).difference(suspendSet))), :]
        return portfolioSuspend, portfolioNonsuspend

    def formatPortfolio(self, asOfDate, country, portfolio):
        adjPortfolio = deepcopy(portfolio)
        adjPortfolio = adjPortfolio.loc[adjPortfolio['USD'] != 0, :]
        adjPortfolio.loc[adjPortfolio['USD'] > 0, 'RATE'] = 0
        prevDate = self.cc.getPrevDateBefore(asOfDate, country)
        portfolioLC = adjPortfolio['USD'].div(self.cdp.getSpot(prevDate, country, type='close'))
        adjPortfolio = pd.concat([portfolioLC.to_frame('LC'), adjPortfolio], axis=1)
        adjPortfolio = pd.merge(self.instrumentMapping[country].set_index(self.tickerType), adjPortfolio, how='right', left_index=True, right_index=True)
        adjPortfolio.index.name = self.tickerType
        adjPortfolio.sort_index(inplace=True)
        return adjPortfolio

    def getBeta(self, asOfDate, country, positionType):
        # betaType = 'PBETAWLD' if positionType == 'risk_apac' else 'PBETALOC'
        # if self.useCache:
        #     loadDate = self.dl.getMappingLoadingDate(asOfDate, tickerType='barra')
        #     data = self.dl.loadData(betaType, loadDate, loadDate, country)
        # else:
        #     data = self.dl.loadBeta(asOfDate, country, self.tickerType)[betaType]
        PBETAfile = r'H:\Shared Folder\Keyi\PBETA\%s\%s_PBETA_%s.csv' % (country,country,asOfDate.strftime('%Y%m%d'))
        data = pd.read_csv(PBETAfile, index_col =[0])
        tickermap = self.tm.getAsOfDateFromSEDOLMapping(asOfDate, country, 'sedol')
        tickermap['id'] = tickermap.index
        tickermap = tickermap.set_index('SEDOL')
        data['id'] = tickermap.loc[data.index]
        data = data.set_index('id')['PBETA'].astype(np.float64)
        return data

    def getRiskPortfolio(self, asOfDate, country, positionType, backtestVersion=''):
        if self.tdt.isRebalanceDay(asOfDate, country):
            portfolio = self.getUnitPortfolio(asOfDate, country, positionType, backtestVersion=backtestVersion)
        else:
            portfolio = self.getReevaluatedPrevPortfolio(asOfDate, country, positionType, backtestVersion=backtestVersion)
        return portfolio

    def countryPerDayRun(self, asOfDate, country):
        positionType = 'risk_apac'
        self.loadRiskData(asOfDate, country)
        for backtestVersion in self.backtestList:
            portfolio = self.getRiskPortfolio(asOfDate, country, positionType, backtestVersion=backtestVersion)
            portfolioFormatted = self.formatPortfolio(asOfDate, country, portfolio)
            self.save(portfolioFormatted, asOfDate, country, positionType, backtestVersion=backtestVersion)

    def loadDataForCountry(self, country):
        print(dt.datetime.now(), country + ' Data collect starts for positionGenerator, %s' % dt.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S'))
        price_df = self.dl.getFilteredMarketData(self.preSDate, self.preEDate, self.tickersByCountry[country], self.tickerType,
                                                 ['CLOSE', 'VOLUME', 'OPEN', 'VWAP'])
        print(dt.datetime.now(), 'price finish')
        adj_df_today = self.dl.getFilteredCAData(self.preSDate, self.preEDate, self.tickersByCountry[country], self.tickerType, ['ADJFACTOR_today'])
        adj_df = adj_df_today['ADJFACTOR_today'].cumprod(axis=0)

        print(dt.datetime.now(), 'finish CA')
        derived_df = self.dl.getFilteredDerivedData(self.preSDate, self.preEDate, self.tickersByCountry[country], self.tickerType, ['MKT_CAP_FLOAT_ONLY',])
        print(dt.datetime.now(), 'finish derived')
        print(dt.datetime.now(),
            country + ' Data collect for positionGenerator is completed, %s' % dt.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'))
        for field in ['VOLUME', 'CLOSE']:
            self.dataFeeders[(country, field)] = dataFeeder(country, field, price_df[field])

        # todo: add 'MKT_CAP_FLOAT_ONLY' back to the field list!
        for field in ['MKT_CAP_FLOAT_ONLY', ]:
            self.dataFeeders[(country, field)] = dataFeeder(country, field, derived_df[field])

        closeBWDByCountry = price_df['CLOSE'].multiply(adj_df)
        executionBWDByCountry = price_df[self.executionType].multiply(adj_df)

        self.dataFeeders[(country, 'CLOSE_BWD')] = dataFeeder(country, 'CLOSE_BWD', closeBWDByCountry)
        pnlDataByCountry = {'CLOSE_BWD': closeBWDByCountry, 'EXECUTION_BWD': executionBWDByCountry}
        self.pnlc.updateCacheData(country, pnlDataByCountry)

    def initializeLoadData(self, country):
        if country in self.countries:
            self.initializeByCountry(country)
            if not self.useCache:
                self.loadDataForCountry(country)

    # @retry(wait=wait_random(1, 3), stop=stop_after_attempt(5), after=after_retry_callback)
    def multiProcessRunPerCountry(self, country):
        print(dt.datetime.now(), country, self.sDate, self.eDate)
        self.initializeLoadData(country)
        dates = self.cc.getDatesBetweenInclusive(self.sDate, self.eDate, country)
        for asOfDate in dates:
            self.countryPerDayRun(asOfDate, country)

    def sequentialRun(self):
        for country in self.countries:
            self.initializeLoadData(country)
        dates = self.cc.getDatesBetweenInclusive(self.sDate, self.eDate, 'APAC')
        for asOfDate in dates:
            for country in self.countries:
                if self.cc.isTradingDay(asOfDate, country):
                    self.countryPerDayRun(asOfDate, country)

    def run(self):
        try:
            print(dt.datetime.now(), 'Running riskAPACRun  multiprocessing {}'.format(self.multiProcessMode))
            if self.multiProcessMode:
                countries = self.countries
                nbProcess = len(countries)
                with mp.Pool(nbProcess) as pool:
                    pool.map(parallel_call, self.prepare_call(__name__, 'multiProcessRunPerCountry', countries))
            else:
                self.sequentialRun()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(dt.datetime.now(), 'Exception occured as {}'.format(e))
            return

    def requestRun(self, adjustByShortSellResponse=False):
        self.setRequestSettings(adjustByShortSellResponse)
        self.resetNbRequestResponseCache()
        self.run()
        self.resetRequestSettings()


if __name__ == '__main__':
    # eDate = dt.datetime.combine(dt.datetime.today().date(), dt.datetime.min.time()) - dt.timedelta(1)
    # sDate = eDate
    sDate = dt.datetime(2007, 5, 8)
    # sDate = dt.datetime(2007, 5, 9)
    # eDate = dt.datetime(2019, 12, 9)
    eDate = dt.datetime(2007, 5, 12)
    countries = None
    # countries = ['SIN']
    # backtestList = ['v0', 'v1', 'v2', 'v3']
    backtestList = None
    # multiProcessMode = True
    multiProcessMode = False
    broker = 'GS'
    sig = None
    # sig = 'DIV2MC_TTM'
    pg = positionGenerator(
        sDate,
        eDate,
        broker,
        countries=countries,
        # sig=None,
        sig=sig,
        executionType='CLOSE',
        beforeMarketOpen=False,
        imposeResponseCut=True,
        shortSellUnconstraintMode=False,
        shortSellInventoryDate=None,
        includePrevinventoryShortSell=True,
        multiProcessMode=multiProcessMode,
        backupMode=False,
        liveMode=True,
        backtestList=backtestList,
        useID=True,
        # useCache=True,
        useCache=False,
    )
    pg.run()

