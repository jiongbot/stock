import React from 'react';
import styled from 'styled-components';

const Panel = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  padding: 24px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
`;

const Title = styled.h2`
  margin: 0 0 20px 0;
  font-size: 20px;
  color: #fff;
`;

const PredictionCard = styled.div`
  background: ${props => props.direction === 'up' 
    ? 'linear-gradient(135deg, rgba(38, 166, 154, 0.2), rgba(38, 166, 154, 0.05))' 
    : 'linear-gradient(135deg, rgba(239, 83, 80, 0.2), rgba(239, 83, 80, 0.05))'};
  border: 2px solid ${props => props.direction === 'up' ? '#26a69a' : '#ef5350'};
  border-radius: 12px;
  padding: 24px;
  text-align: center;
  margin-bottom: 20px;
`;

const DirectionText = styled.div`
  font-size: 32px;
  font-weight: 700;
  color: ${props => props.direction === 'up' ? '#26a69a' : '#ef5350'};
  margin-bottom: 8px;
`;

const ConfidenceText = styled.div`
  font-size: 48px;
  font-weight: 800;
  background: linear-gradient(90deg, #00d4ff, #7b2cbf);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const ConfidenceLabel = styled.div`
  font-size: 14px;
  color: rgba(255, 255, 255, 0.6);
  margin-top: 4px;
`;

const InfoGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 20px;
`;

const InfoItem = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 12px;
`;

const InfoLabel = styled.div`
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 4px;
`;

const InfoValue = styled.div`
  font-size: 16px;
  font-weight: 600;
  color: #fff;
`;

const RefreshButton = styled.button`
  width: 100%;
  padding: 14px;
  margin-top: 20px;
  border: none;
  border-radius: 8px;
  background: linear-gradient(90deg, #00d4ff, #7b2cbf);
  color: white;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0, 212, 255, 0.4);
  }
  
  &:active {
    transform: translateY(0);
  }
`;

const ModelInfo = styled.div`
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const ModelTitle = styled.div`
  font-size: 14px;
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 8px;
`;

const ModelStats = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: 13px;
`;

const StatItem = styled.div`
  text-align: center;
`;

const StatValue = styled.div`
  font-weight: 600;
  color: ${props => props.color || '#fff'};
`;

const StatLabel = styled.div`
  color: rgba(255, 255, 255, 0.5);
  font-size: 11px;
`;

function PredictionPanel({ prediction, onRefresh }) {
  if (!prediction) {
    return (
      <Panel>
        <Title>预测面板</Title>
        <div style={{ textAlign: 'center', padding: '40px', color: 'rgba(255,255,255,0.5)' }}>
          暂无预测数据
        </div>
      </Panel>
    );
  }

  const { prediction: pred, model, current_price, datetime } = prediction;
  const directionText = pred.direction === 'up' ? '📈 上涨' : '📉 下跌';

  return (
    <Panel>
      <Title>🔮 下一周期预测</Title>
      
      <PredictionCard direction={pred.direction}>
        <DirectionText direction={pred.direction}>
          {directionText}
        </DirectionText>
        <ConfidenceText>
          {(pred.confidence * 100).toFixed(1)}%
        </ConfidenceText>
        <ConfidenceLabel>置信度</ConfidenceLabel>
      </PredictionCard>

      <InfoGrid>
        <InfoItem>
          <InfoLabel>当前价格</InfoLabel>
          <InfoValue>${current_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</InfoValue>
        </InfoItem>
        <InfoItem>
          <InfoLabel>上涨概率</InfoLabel>
          <InfoValue style={{ color: '#26a69a' }}>{(pred.probability_up * 100).toFixed(1)}%</InfoValue>
        </InfoItem>
        <InfoItem>
          <InfoLabel>下跌概率</InfoLabel>
          <InfoValue style={{ color: '#ef5350' }}>{(pred.probability_down * 100).toFixed(1)}%</InfoValue>
        </InfoItem>
        <InfoItem>
          <InfoLabel>更新时间</InfoLabel>
          <InfoValue style={{ fontSize: '13px' }}>
            {new Date(datetime).toLocaleTimeString('zh-CN')}
          </InfoValue>
        </InfoItem>
      </InfoGrid>

      <ModelInfo>
        <ModelTitle>模型信息: {model.name}</ModelTitle>
        <ModelStats>
          <StatItem>
            <StatValue color={model.accuracy >= 0.85 ? '#26a69a' : '#ffa726'}>
              {(model.accuracy * 100).toFixed(1)}%
            </StatValue>
            <StatLabel>准确率</StatLabel>
          </StatItem>
          <StatItem>
            <StatValue>{(model.precision * 100).toFixed(1)}%</StatValue>
            <StatLabel>精确率</StatLabel>
          </StatItem>
          <StatItem>
            <StatValue>{(model.recall * 100).toFixed(1)}%</StatValue>
            <StatLabel>召回率</StatLabel>
          </StatItem>
        </ModelStats>
      </ModelInfo>

      <RefreshButton onClick={onRefresh}>
        🔄 刷新预测
      </RefreshButton>
    </Panel>
  );
}

export default PredictionPanel;
