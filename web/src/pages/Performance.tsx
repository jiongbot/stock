import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  LinearProgress,
  Chip,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  Assessment,
  CheckCircle,
  Error,
} from '@mui/icons-material';
import { fetchPerformance } from '../services/api';

const Performance = () => {
  const [performance, setPerformance] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadPerformance();
  }, []);

  const loadPerformance = async () => {
    try {
      const result = await fetchPerformance();
      setPerformance(result.performance || []);
    } catch (e) {
      setError('Failed to load performance data');
    } finally {
      setLoading(false);
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 0.85) return '#4caf50';
    if (accuracy >= 0.70) return '#ff9800';
    return '#f44336';
  };

  const getAccuracyStatus = (accuracy) => {
    if (accuracy >= 0.85) return { label: '优秀', icon: <CheckCircle />, color: 'success' };
    if (accuracy >= 0.70) return { label: '良好', icon: <Assessment />, color: 'warning' };
    return { label: '需改进', icon: <Error />, color: 'error' };
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" py={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        模型性能
      </Typography>

      <Grid container spacing={3}>
        {performance.map((perf) => {
          const status = getAccuracyStatus(perf.accuracy);
          
          return (
            <Grid item xs={12} md={6} key={perf.symbol}>
              <Card sx={{ backgroundColor: '#1a1f3a', height: '100%' }}>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                    <Typography variant="h6">
                      {perf.symbol}
                    </Typography>
                    <Chip
                      icon={status.icon}
                      label={status.label}
                      color={status.color}
                      size="small"
                    />
                  </Box>

                  <Box mb={3}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      准确率
                    </Typography>
                    <Box display="flex" alignItems="center">
                      <Box flexGrow={1} mr={2}>
                        <LinearProgress
                          variant="determinate"
                          value={perf.accuracy * 100}
                          sx={{
                            height: 10,
                            borderRadius: 5,
                            backgroundColor: 'rgba(255,255,255,0.1)',
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: getAccuracyColor(perf.accuracy),
                            },
                          }}
                        />
                      </Box>
                      <Typography variant="h6" sx={{ color: getAccuracyColor(perf.accuracy) }}>
                        {(perf.accuracy * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>

                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        精确率
                      </Typography>
                      <Typography variant="h6">
                        {(perf.precision * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        召回率
                      </Typography>
                      <Typography variant="h6">
                        {(perf.recall * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        F1分数
                      </Typography>
                      <Typography variant="h6">
                        {(perf.f1_score * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        模型类型
                      </Typography>
                      <Typography variant="h6" sx={{ textTransform: 'uppercase' }}>
                        {perf.model_type}
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {performance.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          暂无性能数据。请先训练模型。
        </Alert>
      )}

      <Card sx={{ mt: 3, backgroundColor: '#1a1f3a' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            性能说明
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            <strong>准确率 (Accuracy):</strong> 预测正确的比例。目标 ≥ 85%
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            <strong>精确率 (Precision):</strong> 预测为上涨中实际上涨的比例
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            <strong>召回率 (Recall):</strong> 实际上涨中被正确预测的比例
          </Typography>
          <Typography variant="body2" color="text.secondary">
            <strong>F1分数:</strong> 精确率和召回率的调和平均数
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Performance;
