import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'platform_checker.dart' show getPlatformType;

class ApiService {
  // Get base URL based on platform
  // For web: use localhost
  // For Android emulator: use 10.0.2.2 (maps to host's localhost)
  // For iOS simulator: use localhost
  static String getBaseUrl() {
    // Web platform - always use localhost
    if (kIsWeb) {
      return 'http://localhost:8000';
    }
    
    // Mobile and desktop platforms
    final platform = getPlatformType();
    if (platform == 'android') {
      return 'http://10.0.2.2:8000';  // Android emulator special IP
    } else if (platform == 'ios') {
      return 'http://localhost:8000';
    } else if (platform == 'windows' || platform == 'linux' || platform == 'macos') {
      return 'http://localhost:8000';  // Desktop platforms use localhost
    }
    
    // Default fallback
    return 'http://localhost:8000';
  }

  // Agent Screener endpoint
  static Future<Map<String, dynamic>> searchStocks({
    required String query,
    bool showReasoning = true,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('${getBaseUrl()}/agent'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'query': query,
          'show_reasoning': showReasoning,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to search stocks: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error connecting to API: $e');
    }
  }

  // Chart Analysis endpoint
  static Future<Map<String, dynamic>> analyzeChart({
    required String imageBase64,
    String? additionalContext,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('${getBaseUrl()}/analyze-chart'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'image_base64': imageBase64,
          'additional_context': additionalContext,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to analyze chart: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error analyzing chart: $e');
    }
  }

  // Compare Stock endpoint
  static Future<Map<String, dynamic>> compareStock({
    required String symbol,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('${getBaseUrl()}/compare'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'symbol': symbol,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to compare stock: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error comparing stock: $e');
    }
  }
}
