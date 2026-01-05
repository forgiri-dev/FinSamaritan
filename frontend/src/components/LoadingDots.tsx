import React, { useEffect, useState } from 'react';
import './LoadingDots.css';

/**
 * LoadingDots Component
 * 
 * Animated loading indicator with three dots
 */
export const LoadingDots: React.FC = () => {
  const [dot1Opacity, setDot1Opacity] = useState(0.3);
  const [dot2Opacity, setDot2Opacity] = useState(0.3);
  const [dot3Opacity, setDot3Opacity] = useState(0.3);

  useEffect(() => {
    const animate = (setOpacity: (value: number) => void, delay: number) => {
      const interval = setInterval(() => {
        setOpacity(1);
        setTimeout(() => setOpacity(0.3), 400);
      }, 800);

      // Initial delay
      setTimeout(() => {
        setOpacity(1);
        setTimeout(() => setOpacity(0.3), 400);
      }, delay);

      return interval;
    };

    const interval1 = animate(setDot1Opacity, 0);
    const interval2 = animate(setDot2Opacity, 200);
    const interval3 = animate(setDot3Opacity, 400);

    return () => {
      clearInterval(interval1);
      clearInterval(interval2);
      clearInterval(interval3);
    };
  }, []);

  return (
    <div className="loading-dots-container">
      <span className="loading-dot" style={{ opacity: dot1Opacity }}>●</span>
      <span className="loading-dot" style={{ opacity: dot2Opacity }}>●</span>
      <span className="loading-dot" style={{ opacity: dot3Opacity }}>●</span>
    </div>
  );
};

export default LoadingDots;
