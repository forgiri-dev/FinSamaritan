import React from 'react';
import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import AgentChatScreen from '../screens/AgentChatScreen';

export type RootStackParamList = {
  AgentChat: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

/**
 * App Navigator
 * 
 * Main navigation structure for the app
 */
export const AppNavigator: React.FC = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="AgentChat"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#007AFF',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        <Stack.Screen
          name="AgentChat"
          component={AgentChatScreen}
          options={{
            title: 'FinSights AI',
            headerBackTitle: '',
          }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator;

