import React from 'react';
import {SafeAreaView, StyleSheet, StatusBar} from 'react-native';
import {AppNavigator} from './src/navigation/AppNavigator';

/**
 * FinSights App
 * 
 * The Hybrid Agentic Financial Platform
 * - Cloud Hive (Backend): Manager Agent (Gemini) routes to 7 specialized tools
 * - Edge Sentinel (Frontend): Offline Neural Network filters visual data
 */
const App: React.FC = () => {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#007AFF" />
      <AppNavigator />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
});

export default App;

