import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  CircularProgress,
  Chip,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Refresh,
  Speed,
  Assessment,
} from '@mui/icons-material';
import { fetchPrediction, fetchPerformance } from '../services/api';

const Dashboard = () => {
  const [predictions, setPredictions] = useState({});
  const [performance, setPerformance] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const symbols = ['ETH/USDT', 'BTC/USDT'];

  const loadData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // 获取预测
      const predResults = {};
      for (const symbol of symbols) {
        try {
          const result = await fetchPrediction(symbol);
          predResults[symbol] = result;
        } catch (e) {
          console.error(`Failed to fetch prediction for ${symbol}:`, e);
          predResults[symbol] = null;
        }
      }
      setPredictions(predResults);

      // 获取性能指标
      const perfResult = await fetchPerformance();
      const perfMap = {};
      perfResult.performance?.forEach(p => {
        perfMap[p.symbol] = p;
      });
      setPerformance(perfMap);
    } catch (e) {
      setError('Failed to load data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 60000); // 每分钟刷新
    return () => clearInterval(interval);
  }, []);

  const getConfidenceClass = (confidence) => {
    return `confidence-${confidence || 'low'}`;
  };

  const renderPredictionCard = (symbol) => {
    const pred = predictions[symbol];
    const perf = performance[symbol];

    if (!pred) {
      return (
        <Card sx={{ height: '100%', backgroundColor: '#1a1f3a' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {symbol}
            </Typography>
            <Typography color="text.secondary">
              暂无预测数据
            </Typography>
          </CardContent>
        </Card>
      );
    }

    const isUp = pred.prediction === 1;
    const icon = isUp ? <TrendingUp /> : <TrendingDown />;
    const color = isUp ? '#4caf50' : '#f44336';

    return (
      <Card sx={{ height: '100%', backgroundColor: '#1a1f3a' }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">
              {symbol}
            </Typography>
            <Chip
              label={pred.confidence === 'high' ? '高置信度' : pred.confidence === 'medium' ? '中置信度' : '低置信度'}
              className={getConfidenceClass(pred.confidence)}
              size="small"
            />
          </Box>

          <Box display="flex" alignItems="center" mb={2}>
            <Box sx={{ color, mr: 1 }}>{icon}</Box>
            <Typography variant="h4" sx={{ color }}>
              {isUp ? '上涨' : '下跌'}
            </Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" gutterBottom>
            当前价格: ${pred.current_price?.toLocaleString()}
          </Typography>

          <Typography variant="body2" color="text.secondary" gutterBottom>
            置信度: {(pred.probability * 100).toFixed(1)}%
          </Typography>

          {perf && (
            <Box mt={2} pt={2} borderTop="1px solid rgba(255,255,255,0.1)">
              <Typography variant="body2" color="text.secondary">
                模型准确率: {(perf.accuracy * 100).toFixed(1)}%
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    );
  };

  if (loading && Object.keys(predictions).length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5">
          实时预测
        </Typography>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={loadData}
          disabled={loading}
        >
          刷新
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {symbols.map((symbol) => (
          <Grid item xs={12} md={6} key={symbol}>
            {renderPredictionCard(symbol)}
          </Grid>
        ))}
      </Grid>

      <Box mt={4}>
        <Card sx={{ backgroundColor: '#1a1f3a' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              <Speed sx={{ mr: 1, verticalAlign: 'middle' }} />
              系统状态
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Typography variant="body2" color="text.secondary">
                  预测周期: 15分钟
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="body2" color="text.secondary">
                  支持交易对: ETH/USDT, BTC/USDT
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="body2" color="text.secondary">
                  模型类型: XGBoost
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default Dashboard;
