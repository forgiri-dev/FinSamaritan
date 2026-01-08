import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import '../services/api_service.dart';

class CompareScreen extends StatefulWidget {
  const CompareScreen({super.key});

  @override
  State<CompareScreen> createState() => _CompareScreenState();
}

class _CompareScreenState extends State<CompareScreen> {
  final TextEditingController _symbolController = TextEditingController();
  bool _isLoading = false;
  Map<String, dynamic>? _response;
  String? _error;

  // Demo symbols for quick access
  final List<String> _demoSymbols = [
    'RELIANCE',
    'TCS',
    'HDFCBANK',
    'INFY',
    'ICICIBANK',
  ];

  Future<void> _compareStock() async {
    if (_symbolController.text.trim().isEmpty) {
      setState(() {
        _error = 'Please enter a stock symbol';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
      _response = null;
    });

    try {
      final result = await ApiService.compareStock(
        symbol: _symbolController.text.trim().toUpperCase(),
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

  void _useDemoSymbol(String symbol) {
    _symbolController.text = symbol;
    _compareStock();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Peer Comparison'),
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // Input Section
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.grey.shade100,
            child: Column(
              children: [
                TextField(
                  controller: _symbolController,
                  decoration: InputDecoration(
                    hintText: 'Enter stock symbol (e.g., RELIANCE)',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    filled: true,
                    fillColor: Colors.white,
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.search),
                      onPressed: _isLoading ? null : _compareStock,
                    ),
                  ),
                  textCapitalization: TextCapitalization.characters,
                  onSubmitted: (_) => _compareStock(),
                ),
                const SizedBox(height: 12),
                // Demo Symbol Buttons
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: _demoSymbols.map((symbol) {
                    return ElevatedButton(
                      onPressed: _isLoading
                          ? null
                          : () => _useDemoSymbol(symbol),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green.shade50,
                        foregroundColor: Colors.green.shade900,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 8,
                        ),
                      ),
                      child: Text(
                        symbol,
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
                        Text('Analyzing stock... fetching news...'),
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
                                  Icons.compare_arrows,
                                  size: 64,
                                  color: Colors.grey,
                                ),
                                SizedBox(height: 16),
                                Text(
                                  'Enter a stock symbol to analyze',
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
    final fundamentals = _response!['fundamentals'] as Map<String, dynamic>? ?? {};
    final liveData = _response!['live_data'] as Map<String, dynamic>? ?? {};
    final analysis = _response!['analysis'] as String? ?? '';
    final symbol = _response!['symbol'] as String? ?? 'N/A';

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Stock Header
          Card(
            color: Colors.green.shade50,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    symbol,
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    fundamentals['Name']?.toString() ?? 'N/A',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.grey.shade600,
                    ),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // Fundamentals Table
          Card(
            elevation: 2,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Fundamentals',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _buildFundamentalsTable(fundamentals, liveData),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // Comprehensive Analysis
          if (analysis.isNotEmpty)
            Card(
              color: Colors.green.shade50,
              elevation: 2,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Comprehensive Analysis',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    MarkdownBody(data: analysis),
                  ],
                ),
              ),
            ),

          // Citations (if available)
          if (_response!['citations'] != null &&
              (_response!['citations'] as List).isNotEmpty) ...[
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Sources',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    ...(_response!['citations'] as List).map((citation) {
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Text(
                          '• $citation',
                          style: TextStyle(color: Colors.blue.shade700),
                        ),
                      );
                    }),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildFundamentalsTable(
    Map<String, dynamic> fundamentals,
    Map<String, dynamic> liveData,
  ) {
    final rows = [
      ['Sector', fundamentals['Sector']?.toString() ?? 'N/A'],
      [
        'Current Price',
        liveData['current_price']?.toString() ??
            fundamentals['Current_Price']?.toString() ??
            'N/A'
      ],
      ['PE Ratio', fundamentals['PE_Ratio']?.toString() ?? 'N/A'],
      ['PB Ratio', fundamentals['PB_Ratio']?.toString() ?? 'N/A'],
      ['Sales Growth', '${fundamentals['Sales_Growth']?.toString() ?? 'N/A'}%'],
      ['Profit Margin', '${fundamentals['Profit_Margin']?.toString() ?? 'N/A'}%'],
      ['Market Cap', fundamentals['Market_Cap']?.toString() ?? 'N/A'],
      [
        '52W High',
        liveData['52_week_high']?.toString() ??
            fundamentals['52W_High']?.toString() ??
            'N/A'
      ],
      [
        '52W Low',
        liveData['52_week_low']?.toString() ??
            fundamentals['52W_Low']?.toString() ??
            'N/A'
      ],
    ];

    return Table(
      columnWidths: const {
        0: FlexColumnWidth(2),
        1: FlexColumnWidth(3),
      },
      children: rows.map((row) {
        return TableRow(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: Text(
                row[0],
                style: const TextStyle(fontWeight: FontWeight.w500),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: Text(row[1]),
            ),
          ],
        );
      }).toList(),
    );
  }

  @override
  void dispose() {
    _symbolController.dispose();
    super.dispose();
  }
}

