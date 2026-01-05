import React from 'react';
import {View, StyleSheet, ViewStyle} from 'react-native';
import {Bubble, IMessage} from 'react-native-gifted-chat';
import {MarkdownView} from './MarkdownView';

interface ChatBubbleProps {
  message: IMessage;
  position: 'left' | 'right';
}

/**
 * Custom ChatBubble Component
 * 
 * Wraps GiftedChat's Bubble component with Markdown rendering support
 */
export const ChatBubble: React.FC<ChatBubbleProps> = ({message, position}) => {
  const bubbleStyle: ViewStyle = {
    ...(position === 'left' ? styles.leftBubble : styles.rightBubble),
  };

  return (
    <Bubble
      {...message}
      wrapperStyle={{
        left: styles.leftWrapper,
        right: styles.rightWrapper,
      }}
      textStyle={{
        left: styles.leftText,
        right: styles.rightText,
      }}
      renderCustomText={() => (
        <View style={bubbleStyle}>
          <MarkdownView content={message.text || ''} />
        </View>
      )}
    />
  );
};

const styles = StyleSheet.create({
  leftBubble: {
    backgroundColor: '#f0f0f0',
    borderRadius: 18,
    padding: 10,
    maxWidth: '85%',
  },
  rightBubble: {
    backgroundColor: '#007AFF',
    borderRadius: 18,
    padding: 10,
    maxWidth: '85%',
  },
  leftWrapper: {
    marginBottom: 4,
  },
  rightWrapper: {
    marginBottom: 4,
  },
  leftText: {
    color: '#333',
  },
  rightText: {
    color: '#fff',
  },
});

export default ChatBubble;

