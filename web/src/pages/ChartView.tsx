import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
} from '@mui/material';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import { fetchKlines } from '../services/api';
import { format } from 'date-fns';

const ChartView = () => {
  const [symbol, setSymbol] = useState('ETH/USDT');
  const [klines, setKlines] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadKlines = async () => {
    setLoading(true);
    try {
      const result = await fetchKlines(symbol, '15m', 200);
      const formatted = result.data.map(k => ({
        ...k,
        time: format(new Date(k.datetime), 'MM-dd HH:mm'),
        range: k.high - k.low,
      }));
      setKlines(formatted);
    } catch (e) {
      console.error('Failed to fetch klines:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadKlines();
  }, [symbol]);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5">
          K线图表
        </Typography>
        <FormControl sx={{ minWidth: 150 }}>
          <InputLabel>交易对</InputLabel>
          <Select
            value={symbol}
            label="交易对"
            onChange={(e) => setSymbol(e.target.value)}
          >
            <MenuItem value="ETH/USDT">ETH/USDT</MenuItem>
            <MenuItem value="BTC/USDT">BTC/USDT</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Card sx={{ backgroundColor: '#1a1f3a' }}>
        <CardContent>
          {loading ? (
            <Box display="flex" justifyContent="center" py={4}>
              <CircularProgress />
            </Box>
          ) : (
            <Box sx={{ height: 500 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={klines}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="time" 
                    tick={{ fill: '#888' }}
                    tickLine={{ stroke: '#888' }}
                    minTickGap={30}
                  />
                  <YAxis 
                    yAxisId="left"
                    tick={{ fill: '#888' }}
                    tickLine={{ stroke: '#888' }}
                    domain={['auto', 'auto']}
                  />
                  <YAxis 
                    yAxisId="right"
                    orientation="right"
                    tick={{ fill: '#888' }}
                    tickLine={{ stroke: '#888' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1a1f3a',
                      border: '1px solid rgba(255,255,255,0.1)',
                    }}
                  />
                  <Legend />
                  <Bar
                    yAxisId="right"
                    dataKey="volume"
                    fill="rgba(25, 118, 210, 0.3)"
                    name="成交量"
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="close"
                    stroke="#4caf50"
                    strokeWidth={2}
                    dot={false}
                    name="收盘价"
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="high"
                    stroke="rgba(76, 175, 80, 0.3)"
                    strokeWidth={1}
                    dot={false}
                    name="最高价"
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="low"
                    stroke="rgba(244, 67, 54, 0.3)"
                    strokeWidth={1}
                    dot={false}
                    name="最低价"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default ChartView;
