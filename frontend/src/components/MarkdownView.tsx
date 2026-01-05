import React from 'react';
import ReactMarkdown from 'react-markdown';
import './MarkdownView.css';

interface MarkdownViewProps {
  content: string;
}

/**
 * MarkdownView Component
 * 
 * Renders markdown content from Gemini responses.
 * Essential for displaying tables, bold text, and formatted portfolio/screener results.
 */
export const MarkdownView: React.FC<MarkdownViewProps> = ({ content }) => {
  return (
    <div className="markdown-container">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
};

export default MarkdownView;
