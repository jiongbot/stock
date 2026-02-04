import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import PriceChart from './components/PriceChart';
import PredictionPanel from './components/PredictionPanel';
import PerformanceDashboard from './components/PerformanceDashboard';
import { fetchPrediction, fetchHistory, fetchPerformance } from './services/api';

const Container = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #fff;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
`;

const Header = styled.header`
  padding: 20px 40px;
  background: rgba(0, 0, 0, 0.3);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Title = styled.h1`
  margin: 0;
  font-size: 24px;
  background: linear-gradient(90deg, #00d4ff, #7b2cbf);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const SymbolSelector = styled.div`
  display: flex;
  gap: 10px;
`;

const SymbolButton = styled.button`
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  background: ${props => props.active ? 'linear-gradient(90deg, #00d4ff, #7b2cbf)' : 'rgba(255,255,255,0.1)'};
  color: white;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.3s;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
  }
`;

const MainContent = styled.main`
  padding: 30px 40px;
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 30px;
  max-width: 1600px;
  margin: 0 auto;
  
  @media (max-width: 1200px) {
    grid-template-columns: 1fr;
  }
`;

const LeftPanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const RightPanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const Card = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  padding: 24px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
`;

const LoadingOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  z-index: 1000;
`;

function App() {
  const [symbol, setSymbol] = useState('BTC/USDT');
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 60000); // 每分钟刷新
    return () => clearInterval(interval);
  }, [symbol]);

  const loadData = async () => {
    try {
      setLoading(true);
      const [predData, histData, perfData] = await Promise.all([
        fetchPrediction(symbol),
        fetchHistory(symbol, 200),
        fetchPerformance()
      ]);
      
      setPrediction(predData);
      setHistory(histData.data || []);
      setPerformance(perfData);
    } catch (error) {
      console.error('加载数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container>
      {loading && <LoadingOverlay>加载中...⏳</LoadingOverlay>}
      
      <Header>
        <Title>🔮 Crypto Predictor</Title>
        <SymbolSelector>
          <SymbolButton 
            active={symbol === 'BTC/USDT'} 
            onClick={() => setSymbol('BTC/USDT')}
          >
            ₿ BTC/USDT
          </SymbolButton>
          <SymbolButton 
            active={symbol === 'ETH/USDT'} 
            onClick={() => setSymbol('ETH/USDT')}
          >
            Ξ ETH/USDT
          </SymbolButton>
        </SymbolSelector>
      </Header>

      <MainContent>
        <LeftPanel>
          <Card>
            <PriceChart 
              data={history} 
              symbol={symbol}
              prediction={prediction}
            />
          </Card>
          
          <Card>
            <PerformanceDashboard performance={performance} />
          </Card>
        </LeftPanel>

        <RightPanel>
          <PredictionPanel 
            prediction={prediction} 
            onRefresh={loadData}
          />
        </RightPanel>
      </MainContent>
    </Container>
  );
}

export default App;
