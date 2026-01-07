import React, { useState, useRef } from 'react';
import '../App.css';
import apiService from '../services/api';

const ImageUpload: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      setIsOpen(true);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setAnalysis(null);

    try {
      const result = await apiService.analyzeImage(selectedFile);
      setAnalysis(result);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Failed to analyze image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setIsOpen(false);
    setSelectedFile(null);
    setPreview(null);
    setAnalysis(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <>
      <input
        type="file"
        ref={fileInputRef}
        accept="image/*"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />

      <button
        className="image-upload-button"
        onClick={() => fileInputRef.current?.click()}
        title="Upload Candlestick Chart"
      >
        📊
      </button>

      {isOpen && (
        <>
          <div className="modal-overlay" onClick={handleClose} />
          <div className="image-analysis-modal">
            <div className="modal-header">
              <h2>Candlestick Chart Analysis</h2>
              <button className="modal-close-btn" onClick={handleClose}>
                ×
              </button>
            </div>

            <div className="modal-content">
              {preview && (
                <div className="image-preview-container">
                  <img src={preview} alt="Uploaded chart" className="image-preview" />
                </div>
              )}

              {!analysis && !loading && (
                <button
                  className="btn btn-primary"
                  onClick={handleAnalyze}
                  style={{ marginTop: '1rem', width: '100%' }}
                >
                  Analyze with Edge Sentinel & Gemini
                </button>
              )}

              {loading && (
                <div style={{ textAlign: 'center', padding: '2rem' }}>
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>
                    Analyzing chart...
                  </p>
                </div>
              )}

              {analysis && (
                <div className="analysis-results">
                  <div className="card" style={{ marginTop: '1rem' }}>
                    <div className="card-title">Edge Sentinel Detection</div>
                    {analysis.edge_sentinel.error ? (
                      <p style={{ color: 'var(--accent-danger)' }}>
                        {analysis.edge_sentinel.error}
                      </p>
                    ) : (
                      <div>
                        <div style={{ marginBottom: '0.5rem' }}>
                          <strong>Pattern:</strong>{' '}
                          <span className="badge badge-info">
                            {analysis.edge_sentinel.pattern}
                          </span>
                        </div>
                        <div style={{ marginBottom: '0.5rem' }}>
                          <strong>Trend:</strong>{' '}
                          <span className="badge badge-info">
                            {analysis.edge_sentinel.trend}
                          </span>
                        </div>
                        <div style={{ marginBottom: '0.5rem' }}>
                          <strong>Full Classification:</strong>{' '}
                          <span style={{ color: 'var(--text-secondary)' }}>
                            {analysis.edge_sentinel.full_classification}
                          </span>
                        </div>
                        <div>
                          <strong>Confidence:</strong>{' '}
                          <span style={{ color: 'var(--accent-primary)' }}>
                            {analysis.edge_sentinel.confidence}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {analysis.gemini_analysis && (
                    <div className="card" style={{ marginTop: '1rem' }}>
                      <div className="card-title">Gemini AI Analysis</div>
                      <div
                        style={{
                          whiteSpace: 'pre-wrap',
                          lineHeight: '1.6',
                          color: 'var(--text-primary)',
                        }}
                      >
                        {analysis.gemini_analysis}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </>
  );
};

export default ImageUpload;

