import React, {useState, useCallback, useEffect} from 'react';
import {
  View,
  StyleSheet,
  Alert,
  Platform,
  KeyboardAvoidingView,
  StatusBar,
} from 'react-native';
import {GiftedChat, IMessage, InputToolbar, Actions, Composer, Bubble} from 'react-native-gifted-chat';
import {launchImageLibrary, ImagePickerResponse, Asset} from 'react-native-image-picker';
import {sendAgentMessage, analyzeChart} from '../api/agent';
import {scanImage, ChartAnalysis} from '../services/EdgeSentinel';
import {MarkdownView} from '../components/MarkdownView';
import {LoadingDots} from '../components/LoadingDots';

/**
 * AgentChatScreen
 * 
 * Main chat interface where users interact with the FinSights AI agent.
 * Features:
 * - Text-based queries to the agent
 * - Image upload with Edge Sentinel filtering
 * - Markdown rendering for formatted responses
 */
export const AgentChatScreen: React.FC = () => {
  const [messages, setMessages] = useState<IMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Welcome message
    setMessages([
      {
        _id: 1,
        text: '👋 Welcome to FinSights! I\'m your AI Wealth Manager.\n\nI can help you:\n• Manage your portfolio\n• Analyze stocks\n• Screen investments\n• Backtest strategies\n• Compare peers\n• Analyze chart images\n\nHow can I assist you today?',
        createdAt: new Date(),
        user: {
          _id: 2,
          name: 'FinSights AI',
        },
      },
    ]);
  }, []);

  const onSend = useCallback(async (newMessages: IMessage[] = []) => {
    setMessages((previousMessages) =>
      GiftedChat.append(previousMessages, newMessages)
    );

    const userMessage = newMessages[0];
    if (!userMessage.text) return;

    setIsLoading(true);

    try {
      const response = await sendAgentMessage(userMessage.text);
      
      const aiMessage: IMessage = {
        _id: Math.round(Math.random() * 1000000),
        text: response,
        createdAt: new Date(),
        user: {
          _id: 2,
          name: 'FinSights AI',
        },
      };

      setMessages((previousMessages) =>
        GiftedChat.append(previousMessages, [aiMessage])
      );
    } catch (error: any) {
      const errorMessage: IMessage = {
        _id: Math.round(Math.random() * 1000000),
        text: `❌ Error: ${error.message}\n\nPlease make sure the backend server is running.`,
        createdAt: new Date(),
        user: {
          _id: 2,
          name: 'FinSights AI',
        },
      };

      setMessages((previousMessages) =>
        GiftedChat.append(previousMessages, [errorMessage])
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleImagePicker = useCallback(() => {
    const options = {
      mediaType: 'photo' as const,
      quality: 0.8,
      maxWidth: 1024,
      maxHeight: 1024,
      includeBase64: true, // Get base64 directly from image picker
    };

    launchImageLibrary(options, async (response: ImagePickerResponse) => {
      if (response.didCancel || response.errorCode) {
        return;
      }

      const asset = response.assets?.[0];
      if (!asset?.uri) return;

      // Show user that image is being processed
      const processingMessage: IMessage = {
        _id: Math.round(Math.random() * 1000000),
        text: '🖼️ Processing image...',
        createdAt: new Date(),
        user: {
          _id: 1,
        },
      };

      setMessages((previousMessages) =>
        GiftedChat.append(previousMessages, [processingMessage])
      );

      try {
        // Step 1: Edge Sentinel - Analyze chart patterns and trends
        const chartAnalysis = await scanImage(asset.uri);

        if (!chartAnalysis.isChart) {
          // Remove processing message
          setMessages((previousMessages) =>
            previousMessages.filter((msg) => msg._id !== processingMessage._id)
          );

          Alert.alert(
            'Not a Chart',
            'The Edge Sentinel detected this is not a financial chart. Please upload a chart image.',
            [{text: 'OK'}]
          );
          return;
        }

        // Show detected pattern and trend
        if (chartAnalysis.pattern && chartAnalysis.trend) {
          const patternInfo = `📊 Edge Sentinel detected: ${chartAnalysis.pattern} pattern in ${chartAnalysis.trend} trend`;
          setMessages((previousMessages) =>
            previousMessages.map((msg) =>
              msg._id === processingMessage._id
                ? {
                    ...msg,
                    text: `${patternInfo}\n\n🔍 Analyzing chart with Vision Agent...`,
                  }
                : msg
            )
          );
        }

        // Step 2: Get base64 from image picker response
        let base64Image: string;
        if (asset.base64) {
          base64Image = `data:${asset.type || 'image/jpeg'};base64,${asset.base64}`;
        } else {
          // Fallback: if base64 not available, we need to read the file
          // For now, show an error
          throw new Error('Could not read image data. Please try again.');
        }

        // Step 3: Send to backend for analysis
        setIsLoading(true);

        // Update processing message
        setMessages((previousMessages) =>
          previousMessages.map((msg) =>
            msg._id === processingMessage._id
              ? {
                  ...msg,
                  text: '🔍 Analyzing chart with Vision Agent...',
                }
              : msg
          )
        );

        const analysis = await analyzeChart(base64Image);

        // Replace processing message with analysis
        setMessages((previousMessages) =>
          previousMessages.map((msg) =>
            msg._id === processingMessage._id
              ? {
                  ...msg,
                  _id: Math.round(Math.random() * 1000000),
                  text: analysis,
                  user: {
                    _id: 2,
                    name: 'FinSights AI',
                  },
                }
              : msg
          )
        );
      } catch (error: any) {
        // Remove processing message
        setMessages((previousMessages) =>
          previousMessages.filter((msg) => msg._id !== processingMessage._id)
        );

        const errorMessage: IMessage = {
          _id: Math.round(Math.random() * 1000000),
          text: `❌ Error analyzing chart: ${error.message}`,
          createdAt: new Date(),
          user: {
            _id: 2,
            name: 'FinSights AI',
          },
        };

        setMessages((previousMessages) =>
          GiftedChat.append(previousMessages, [errorMessage])
        );
      } finally {
        setIsLoading(false);
      }
    });
  }, []);

  const renderBubble = useCallback((props: any) => {
    return (
      <Bubble
        {...props}
        wrapperStyle={{
          left: styles.leftBubbleWrapper,
          right: styles.rightBubbleWrapper,
        }}
        textStyle={{
          left: styles.leftBubbleText,
          right: styles.rightBubbleText,
        }}
      />
    );
  }, []);

  const renderMessageText = useCallback((props: any) => {
    return (
      <View style={styles.messageTextContainer}>
        <MarkdownView content={props.currentMessage?.text || ''} />
      </View>
    );
  }, []);

  const renderInputToolbar = useCallback((props: any) => {
    return (
      <InputToolbar
        {...props}
        containerStyle={styles.inputToolbar}
        primaryStyle={styles.inputPrimary}
      />
    );
  }, []);

  const renderComposer = useCallback((props: any) => {
    return (
      <Composer
        {...props}
        textInputStyle={styles.composer}
        placeholder="Ask about stocks, portfolio, or upload a chart..."
      />
    );
  }, []);

  const renderActions = useCallback((props: any) => {
    return (
      <Actions
        {...props}
        containerStyle={styles.actionsContainer}
        icon={() => (
          <View style={styles.imageButton}>
            <View style={styles.imageIcon} />
          </View>
        )}
        onPressActionButton={handleImagePicker}
      />
    );
  }, [handleImagePicker]);

  const renderLoading = useCallback(() => {
    if (!isLoading) return null;
    return (
      <View style={styles.loadingContainer}>
        <LoadingDots />
      </View>
    );
  }, [isLoading]);

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <StatusBar barStyle="dark-content" />
      <GiftedChat
        messages={messages}
        onSend={onSend}
        user={{
          _id: 1,
        }}
        renderBubble={renderBubble}
        renderMessageText={renderMessageText}
        renderInputToolbar={renderInputToolbar}
        renderComposer={renderComposer}
        renderActions={renderActions}
        renderLoading={renderLoading}
        placeholder="Ask about stocks, portfolio, or upload a chart..."
        alwaysShowSend
        scrollToBottom
        showUserAvatar={false}
        showAvatarForEveryMessage={false}
        minInputToolbarHeight={60}
      />
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  leftBubbleWrapper: {
    marginBottom: 4,
  },
  rightBubbleWrapper: {
    marginBottom: 4,
  },
  leftBubbleText: {
    color: '#333',
  },
  rightBubbleText: {
    color: '#fff',
  },
  messageTextContainer: {
    padding: 4,
  },
  inputToolbar: {
    backgroundColor: '#f5f5f5',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  inputPrimary: {
    alignItems: 'center',
  },
  composer: {
    backgroundColor: '#fff',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 16,
    maxHeight: 100,
  },
  actionsContainer: {
    marginRight: 8,
    marginBottom: 4,
  },
  imageButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  imageIcon: {
    width: 20,
    height: 20,
    backgroundColor: '#fff',
    borderRadius: 4,
  },
  loadingContainer: {
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
});

export default AgentChatScreen;

