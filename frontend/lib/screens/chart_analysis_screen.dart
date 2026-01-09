import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import '../services/api_service.dart';

class ChartAnalysisScreen extends StatefulWidget {
  const ChartAnalysisScreen({super.key});

  @override
  State<ChartAnalysisScreen> createState() => _ChartAnalysisScreenState();
}

class _ChartAnalysisScreenState extends State<ChartAnalysisScreen> {
  final ImagePicker _picker = ImagePicker();
  Uint8List? _selectedImageBytes;
  String? _selectedImagePath;
  bool _isLoading = false;
  String? _analysis;
  String? _error;

  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 85,
      );

      if (image != null) {
        final imageBytes = await image.readAsBytes();
        setState(() {
          _selectedImageBytes = imageBytes;
          _selectedImagePath = image.path;
          _analysis = null;
          _error = null;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Error picking image: $e';
      });
    }
  }

  Future<void> _takePhoto() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: 85,
      );

      if (image != null) {
        final imageBytes = await image.readAsBytes();
        setState(() {
          _selectedImageBytes = imageBytes;
          _selectedImagePath = image.path;
          _analysis = null;
          _error = null;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Error taking photo: $e';
      });
    }
  }

  Future<void> _analyzeChart() async {
    if (_selectedImageBytes == null) {
      setState(() {
        _error = 'Please select an image first';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
      _analysis = null;
    });

    try {
      // Convert image bytes to base64
      final base64Image = base64Encode(_selectedImageBytes!);

      final result = await ApiService.analyzeChart(
        imageBase64: base64Image,
      );

      // Build combined analysis from dual-processing results
      String combinedAnalysis = '';
      bool hasAnyAnalysis = false;
      
      // Edge Sentinel Analysis
      if (result['edge_sentinel'] != null) {
        final edgeResult = result['edge_sentinel'];
        if (edgeResult['success'] == true) {
          final topPred = edgeResult['top_prediction'];
          if (topPred != null) {
            combinedAnalysis += '## Edge Sentinel Analysis (Local Model)\n\n';
            combinedAnalysis += '**Pattern:** ${topPred['pattern']}\n';
            combinedAnalysis += '**Trend:** ${topPred['trend']}\n';
            combinedAnalysis += '**Confidence:** ${(topPred['confidence'] * 100).toStringAsFixed(1)}%\n\n';
            hasAnyAnalysis = true;
          } else if (edgeResult['predictions'] != null && (edgeResult['predictions'] as List).isNotEmpty) {
            // Fallback: use first prediction if top_prediction is missing
            final predictions = edgeResult['predictions'] as List;
            if (predictions.isNotEmpty) {
              final firstPred = predictions[0];
              combinedAnalysis += '## Edge Sentinel Analysis (Local Model)\n\n';
              combinedAnalysis += '**Pattern:** ${firstPred['pattern']}\n';
              combinedAnalysis += '**Trend:** ${firstPred['trend']}\n';
              combinedAnalysis += '**Confidence:** ${(firstPred['confidence'] * 100).toStringAsFixed(1)}%\n\n';
              hasAnyAnalysis = true;
            }
          }
        } else {
          // Show error if Edge Sentinel failed
          combinedAnalysis += '## Edge Sentinel Analysis (Local Model)\n\n';
          combinedAnalysis += '❌ **Error:** ${edgeResult['error'] ?? 'Analysis failed'}\n\n';
        }
      }
      
      // Gemini Vision Analysis
      if (result['gemini_vision'] != null) {
        final geminiResult = result['gemini_vision'];
        if (geminiResult['success'] == true) {
          final analysis = geminiResult['analysis'];
          if (analysis != null && analysis.toString().isNotEmpty) {
            combinedAnalysis += '## Gemini Vision Analysis\n\n';
            combinedAnalysis += analysis.toString();
            combinedAnalysis += '\n\n';
            hasAnyAnalysis = true;
          }
        } else {
          // Show error if Gemini Vision failed
          combinedAnalysis += '## Gemini Vision Analysis\n\n';
          combinedAnalysis += '❌ **Error:** ${geminiResult['error'] ?? 'Analysis failed'}\n\n';
        }
      }
      
      // Combined Summary
      if (result['combined_summary'] != null && result['combined_summary'].toString().isNotEmpty) {
        combinedAnalysis += '## Combined Analysis\n\n';
        combinedAnalysis += result['combined_summary'];
        hasAnyAnalysis = true;
      }

      setState(() {
        if (combinedAnalysis.isNotEmpty) {
          _analysis = combinedAnalysis;
        } else {
          _error = 'No analysis results available. Both models may have failed.';
        }
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chart Analysis'),
        backgroundColor: Colors.purple.shade700,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Picker Section
            Card(
              elevation: 2,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    const Text(
                      'Upload Trading Chart',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton.icon(
                          onPressed: _pickImage,
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Gallery'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.purple.shade100,
                            foregroundColor: Colors.purple.shade900,
                          ),
                        ),
                        ElevatedButton.icon(
                          onPressed: _takePhoto,
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Camera'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.purple.shade100,
                            foregroundColor: Colors.purple.shade900,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Selected Image Display
            if (_selectedImageBytes != null) ...[
              Card(
                elevation: 2,
                child: Column(
                  children: [
                    kIsWeb
                        ? Image.memory(
                            _selectedImageBytes!,
                            fit: BoxFit.contain,
                            height: 300,
                          )
                        : Image.memory(
                            _selectedImageBytes!,
                            fit: BoxFit.contain,
                            height: 300,
                          ),
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: ElevatedButton(
                        onPressed: _isLoading ? null : _analyzeChart,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.purple.shade700,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 32,
                            vertical: 16,
                          ),
                        ),
                        child: _isLoading
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    Colors.white,
                                  ),
                                ),
                              )
                            : const Text(
                                'Analyze Chart',
                                style: TextStyle(fontSize: 16),
                              ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
            ],

            // Error Display
            if (_error != null)
              Card(
                color: Colors.red.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      Icon(Icons.error_outline, color: Colors.red.shade700),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          _error!,
                          style: TextStyle(color: Colors.red.shade700),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

            // Analysis Results
            if (_analysis != null) ...[
              Card(
                color: Colors.purple.shade50,
                elevation: 2,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Technical Analysis',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      MarkdownBody(data: _analysis!),
                    ],
                  ),
                ),
              ),
            ],

            // Empty State
            if (_selectedImageBytes == null && _analysis == null && _error == null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(48),
                  child: Column(
                    children: [
                      Icon(
                        Icons.show_chart,
                        size: 64,
                        color: Colors.grey.shade400,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'Upload a trading chart to get AI-powered technical analysis',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.grey.shade600,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

