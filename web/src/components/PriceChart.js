import React, { useEffect, useRef } from 'react';
import styled from 'styled-components';
import { createChart, ColorType } from 'lightweight-charts';

const ChartContainer = styled.div`
  width: 100%;
  height: 400px;
`;

const ChartTitle = styled.h3`
  margin: 0 0 16px 0;
  font-size: 18px;
  color: #fff;
`;

const PriceInfo = styled.div`
  display: flex;
  gap: 20px;
  margin-bottom: 16px;
  flex-wrap: wrap;
`;

const PriceItem = styled.div`
  display: flex;
  flex-direction: column;
`;

const PriceLabel = styled.span`
  font-size: 12px;
  color: rgba(255, 255, 255, 0.6);
`;

const PriceValue = styled.span`
  font-size: 16px;
  font-weight: 600;
  color: ${props => props.color || '#fff'};
`;

function PriceChart({ data, symbol, prediction }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.1)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.1)' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.2)',
      },
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.2)',
        timeVisible: true,
      },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    // 转换数据格式
    const chartData = data.map(item => ({
      time: item.timestamp / 1000,
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
    }));

    candlestickSeries.setData(chartData);
    chart.timeScale().fitContent();

    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;

    return () => {
      chart.remove();
    };
  }, [data]);

  // 添加预测标记
  useEffect(() => {
    if (!candlestickSeriesRef.current || !prediction || data.length === 0) return;

    const lastCandle = data[data.length - 1];
    const markers = [{
      time: lastCandle.timestamp / 1000,
      position: prediction.prediction.direction === 'up' ? 'belowBar' : 'aboveBar',
      color: prediction.prediction.direction === 'up' ? '#26a69a' : '#ef5350',
      shape: prediction.prediction.direction === 'up' ? 'arrowUp' : 'arrowDown',
      text: `预测${prediction.prediction.direction === 'up' ? '上涨' : '下跌'} ${(prediction.prediction.confidence * 100).toFixed(1)}%`,
    }];

    candlestickSeriesRef.current.setMarkers(markers);
  }, [prediction, data]);

  const latestPrice = data.length > 0 ? data[data.length - 1].close : 0;
  const priceChange = data.length > 1 
    ? ((latestPrice - data[data.length - 2].close) / data[data.length - 2].close * 100)
    : 0;

  return (
    <div>
      <ChartTitle>{symbol} 15分钟K线</ChartTitle>
      
      <PriceInfo>
        <PriceItem>
          <PriceLabel>当前价格</PriceLabel>
          <PriceValue>${latestPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}</PriceValue>
        </PriceItem>
        <PriceItem>
          <PriceLabel>涨跌</PriceLabel>
          <PriceValue color={priceChange >= 0 ? '#26a69a' : '#ef5350'}>
            {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
          </PriceValue>
        </PriceItem>
        <PriceItem>
          <PriceLabel>24h最高</PriceLabel>
          <PriceValue>
            ${Math.max(...data.slice(-96).map(d => d.high)).toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </PriceValue>
        </PriceItem>
        <PriceItem>
          <PriceLabel>24h最低</PriceLabel>
          <PriceValue>
            ${Math.min(...data.slice(-96).map(d => d.low)).toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </PriceValue>
        </PriceItem>
      </PriceInfo>

      <ChartContainer ref={chartContainerRef} />
    </div>
  );
}

export default PriceChart;
