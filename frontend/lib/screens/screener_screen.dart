import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import '../services/api_service.dart';

class ScreenerScreen extends StatefulWidget {
  const ScreenerScreen({super.key});

  @override
  State<ScreenerScreen> createState() => _ScreenerScreenState();
}

class _ScreenerScreenState extends State<ScreenerScreen> {
  final TextEditingController _queryController = TextEditingController();
  bool _isLoading = false;
  Map<String, dynamic>? _response;
  String? _error;

  // Demo queries for quick access
  final List<String> _demoQueries = [
    "Show me undervalued IT stocks",
    "Find high growth banks",
    "Show me cheap pharma stocks",
    "Find stocks with PE < 15",
    "Show me high profit margin companies",
  ];

  Future<void> _searchStocks() async {
    if (_queryController.text.trim().isEmpty) {
      setState(() {
        _error = 'Please enter a search query';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
      _response = null;
    });

    try {
      final result = await ApiService.searchStocks(
        query: _queryController.text.trim(),
        showReasoning: true,
      );

      setState(() {
        _response = result;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  void _useDemoQuery(String query) {
    _queryController.text = query;
    _searchStocks();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Smart Screener'),
        backgroundColor: Colors.blue.shade700,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // Search Input Section
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.grey.shade100,
            child: Column(
              children: [
                TextField(
                  controller: _queryController,
                  decoration: InputDecoration(
                    hintText: 'e.g., "Show me cheap IT stocks"',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    filled: true,
                    fillColor: Colors.white,
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.search),
                      onPressed: _isLoading ? null : _searchStocks,
                    ),
                  ),
                  onSubmitted: (_) => _searchStocks(),
                ),
                const SizedBox(height: 12),
                // Demo Query Buttons
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: _demoQueries.map((query) {
                    return ElevatedButton(
                      onPressed: _isLoading
                          ? null
                          : () => _useDemoQuery(query),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue.shade50,
                        foregroundColor: Colors.blue.shade900,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 8,
                        ),
                      ),
                      child: Text(
                        query,
                        style: const TextStyle(fontSize: 12),
                      ),
                    );
                  }).toList(),
                ),
              ],
            ),
          ),

          // Results Section
          Expanded(
            child: _isLoading
                ? const Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text('Thinking... querying database...'),
                      ],
                    ),
                  )
                : _error != null
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.error_outline,
                              size: 64,
                              color: Colors.red.shade300,
                            ),
                            const SizedBox(height: 16),
                            const Text(
                              'Error',
                              style: TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 32),
                              child: Text(
                                _error!,
                                textAlign: TextAlign.center,
                                style: const TextStyle(color: Colors.red),
                              ),
                            ),
                          ],
                        ),
                      )
                    : _response == null
                        ? const Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  Icons.search,
                                  size: 64,
                                  color: Colors.grey,
                                ),
                                SizedBox(height: 16),
                                Text(
                                  'Enter a query to search stocks',
                                  style: TextStyle(
                                    fontSize: 16,
                                    color: Colors.grey,
                                  ),
                                ),
                              ],
                            ),
                          )
                        : _buildResults(),
          ),
        ],
      ),
    );
  }

  Widget _buildResults() {
    final results = _response!['results'] as List<dynamic>? ?? [];
    final summary = _response!['summary'] as String? ?? '';
    final reasoning = _response!['reasoning'] as String?;
    final count = _response!['count'] as int? ?? 0;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Summary Section
          if (summary.isNotEmpty)
            Card(
              color: Colors.blue.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Analysis Summary',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    MarkdownBody(data: summary),
                  ],
                ),
              ),
            ),

          const SizedBox(height: 16),

          // Reasoning Trace (if available)
          if (reasoning != null && reasoning.isNotEmpty)
            Card(
              color: Colors.grey.shade100,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Reasoning Trace',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        fontStyle: FontStyle.italic,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      reasoning,
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade700,
                      ),
                    ),
                  ],
                ),
              ),
            ),

          const SizedBox(height: 16),

          // Results Count
          Text(
            'Found $count stocks',
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 12),

          // Stock Cards
          ...results.map((stock) => _buildStockCard(stock)),
        ],
      ),
    );
  }

  Widget _buildStockCard(Map<String, dynamic> stock) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        stock['Symbol']?.toString() ?? 'N/A',
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        stock['Name']?.toString() ?? 'N/A',
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey.shade600,
                        ),
                      ),
                    ],
                  ),
                ),
                Chip(
                  label: Text(
                    stock['Sector']?.toString() ?? 'N/A',
                    style: const TextStyle(fontSize: 12),
                  ),
                  backgroundColor: Colors.blue.shade100,
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildMetric('PE', stock['PE_Ratio']?.toString() ?? 'N/A'),
                _buildMetric('PB', stock['PB_Ratio']?.toString() ?? 'N/A'),
                _buildMetric(
                  'Growth',
                  '${stock['Sales_Growth']?.toString() ?? 'N/A'}%',
                ),
                _buildMetric(
                  'Price',
                  '₹${stock['Current_Price']?.toString() ?? 'N/A'}',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetric(String label, String value) {
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey.shade600,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _queryController.dispose();
    super.dispose();
  }
}

