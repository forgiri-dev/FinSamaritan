import React from 'react';
import {StyleSheet, View} from 'react-native';
import Markdown from 'react-native-markdown-display';

interface MarkdownViewProps {
  content: string;
}

/**
 * MarkdownView Component
 * 
 * Renders markdown content from Gemini responses.
 * Essential for displaying tables, bold text, and formatted portfolio/screener results.
 */
export const MarkdownView: React.FC<MarkdownViewProps> = ({content}) => {
  return (
    <View style={styles.container}>
      <Markdown
        style={markdownStyles}
        mergeStyle={true}
      >
        {content}
      </Markdown>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

const markdownStyles = {
  body: {
    fontFamily: 'System',
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  heading1: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 10,
    marginBottom: 10,
  },
  heading2: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 8,
    marginBottom: 8,
  },
  heading3: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 6,
    marginBottom: 6,
  },
  paragraph: {
    marginTop: 4,
    marginBottom: 4,
  },
  strong: {
    fontWeight: 'bold',
  },
  em: {
    fontStyle: 'italic',
  },
  list_item: {
    marginTop: 4,
    marginBottom: 4,
  },
  table: {
    borderWidth: 1,
    borderColor: '#ddd',
    marginVertical: 8,
  },
  thead: {
    backgroundColor: '#f5f5f5',
  },
  th: {
    padding: 8,
    fontWeight: 'bold',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  td: {
    padding: 8,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  code_inline: {
    backgroundColor: '#f4f4f4',
    padding: 2,
    borderRadius: 3,
    fontFamily: 'monospace',
  },
  code_block: {
    backgroundColor: '#f4f4f4',
    padding: 10,
    borderRadius: 5,
    fontFamily: 'monospace',
  },
  blockquote: {
    borderLeftWidth: 4,
    borderLeftColor: '#ddd',
    paddingLeft: 10,
    marginLeft: 10,
    fontStyle: 'italic',
  },
};

export default MarkdownView;

