import React from 'react';
import styled from 'styled-components';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const Title = styled.h3`
  margin: 0 0 20px 0;
  font-size: 18px;
  color: #fff;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
`;

const StatCard = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  padding: 16px;
  text-align: center;
`;

const StatValue = styled.div`
  font-size: 28px;
  font-weight: 700;
  color: ${props => props.color || '#fff'};
  margin-bottom: 4px;
`;

const StatLabel = styled.div`
  font-size: 13px;
  color: rgba(255, 255, 255, 0.6);
`;

const ChartContainer = styled.div`
  height: 250px;
  margin-top: 20px;
`;

function PerformanceDashboard({ performance }) {
  if (!performance || !performance.models) {
    return (
      <div>
        <Title>模型性能</Title>
        <div style={{ textAlign: 'center', padding: '40px', color: 'rgba(255,255,255,0.5)' }}>
          暂无性能数据
        </div>
      </div>
    );
  }

  const models = performance.models;
  const chartData = Object.entries(models).map(([symbol, stats]) => ({
    symbol: symbol.replace('/USDT', ''),
    accuracy: (stats.accuracy * 100).toFixed(1),
    precision: (stats.precision * 100).toFixed(1),
    recall: (stats.recall * 100).toFixed(1),
    f1: (stats.f1 * 100).toFixed(1),
  }));

  return (
    <div>
      <Title>📊 模型性能指标</Title>
      
      <StatsGrid>
        {Object.entries(models).map(([symbol, stats]) => (
          <React.Fragment key={symbol}>
            <StatCard>
              <StatValue color={stats.accuracy >= 0.85 ? '#26a69a' : '#ffa726'}>
                {(stats.accuracy * 100).toFixed(1)}%
              </StatValue>
              <StatLabel>{symbol.replace('/USDT', '')} 准确率</StatLabel>
            </StatCard>
            <StatCard>
              <StatValue>{(stats.f1 * 100).toFixed(1)}%</StatValue>
              <StatLabel>{symbol.replace('/USDT', '')} F1分数</StatLabel>
            </StatCard>
          </React.Fragment>
        ))}
      </StatsGrid>

      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="symbol" 
              stroke="rgba(255,255,255,0.5)"
              tick={{ fill: 'rgba(255,255,255,0.7)' }}
            />
            <YAxis 
              stroke="rgba(255,255,255,0.5)"
              tick={{ fill: 'rgba(255,255,255,0.7)' }}
              domain={[0, 100]}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(30, 30, 50, 0.95)',
                border: '1px solid rgba(255,255,255,0.2)',
                borderRadius: '8px',
                color: '#fff'
              }}
              formatter={(value) => [`${value}%`]}
            />
            
            <Bar dataKey="accuracy" name="准确率" fill="#00d4ff" radius={[4, 4, 0, 0]} />
            <Bar dataKey="precision" name="精确率" fill="#7b2cbf" radius={[4, 4, 0, 0]} />
            <Bar dataKey="recall" name="召回率" fill="#26a69a" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  );
}

export default PerformanceDashboard;
