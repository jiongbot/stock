import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Tabs,
  Tab,
  Paper,
} from '@mui/material';
import { TrendingUp, Assessment, ShowChart } from '@mui/icons-material';
import Dashboard from './pages/Dashboard';
import ChartView from './pages/ChartView';
import Performance from './pages/Performance';
import './App.css';

function App() {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ backgroundColor: '#1a1f3a' }}>
        <Toolbar>
          <TrendingUp sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Crypto Predictor
          </Typography>
          <Typography variant="body2" color="text.secondary">
            ETH/BTC 15分钟预测
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
        <Paper sx={{ mb: 2 }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            indicatorColor="primary"
            textColor="primary"
            variant="fullWidth"
          >
            <Tab icon={<ShowChart />} label="实时预测" />
            <Tab icon={<Assessment />} label="K线图表" />
            <Tab icon={<TrendingUp />} label="模型性能" />
          </Tabs>
        </Paper>

        <Box sx={{ mt: 2 }}>
          {tabValue === 0 && <Dashboard />}
          {tabValue === 1 && <ChartView />}
          {tabValue === 2 && <Performance />}
        </Box>
      </Container>
    </Box>
  );
}

export default App;
