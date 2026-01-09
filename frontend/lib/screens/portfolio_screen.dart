import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'dart:convert';

class PortfolioScreen extends StatefulWidget {
  const PortfolioScreen({super.key});

  @override
  State<PortfolioScreen> createState() => _PortfolioScreenState();
}

class _PortfolioScreenState extends State<PortfolioScreen> {
  final TextEditingController _searchController = TextEditingController();
  List<Map<String, dynamic>> _searchResults = [];
  List<Map<String, dynamic>> _portfolio = [];
  Map<String, dynamic>? _portfolioAnalysis;
  bool _isSearching = false;
  bool _isLoadingPortfolio = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadPortfolio();
  }

  Future<void> _loadPortfolio() async {
    setState(() {
      _isLoadingPortfolio = true;
      _error = null;
    });

    try {
      final result = await ApiService.getPortfolio();
      if (result['success'] == true) {
        setState(() {
          _portfolioAnalysis = result;
          _portfolio = List<Map<String, dynamic>>.from(result['holdings'] ?? []);
          _isLoadingPortfolio = false;
        });
      } else {
        setState(() {
          _error = result['error'] ?? 'Failed to load portfolio';
          _isLoadingPortfolio = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoadingPortfolio = false;
      });
    }
  }

  Future<void> _searchStocks(String query) async {
    if (query.trim().isEmpty) {
      setState(() {
        _searchResults = [];
      });
      return;
    }

    setState(() {
      _isSearching = true;
      _error = null;
    });

    try {
      final result = await ApiService.searchStocks(query: query);
      if (result['success'] == true) {
        setState(() {
          _searchResults = List<Map<String, dynamic>>.from(result['stocks'] ?? []);
          _isSearching = false;
        });
      } else {
        setState(() {
          _error = result['error'] ?? 'Search failed';
          _isSearching = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isSearching = false;
      });
    }
  }

  Future<void> _addToPortfolio(String symbol, double currentPrice) async {
    // Show dialog to get shares and buy price
    final sharesController = TextEditingController();
    final buyPriceController = TextEditingController(text: currentPrice.toStringAsFixed(2));

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Add $symbol to Portfolio'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: sharesController,
              decoration: const InputDecoration(
                labelText: 'Number of Shares',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 16),
            TextField(
              controller: buyPriceController,
              decoration: const InputDecoration(
                labelText: 'Buy Price (₹)',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.numberWithOptions(decimal: true),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              if (sharesController.text.isNotEmpty && buyPriceController.text.isNotEmpty) {
                Navigator.pop(context, true);
              }
            },
            child: const Text('Add'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      try {
        final shares = int.parse(sharesController.text);
        final buyPrice = double.parse(buyPriceController.text);

        final result = await ApiService.addToPortfolio(
          symbol: symbol,
          shares: shares,
          buyPrice: buyPrice,
        );

        if (result['success'] == true) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(result['message'] ?? 'Added to portfolio')),
          );
          _loadPortfolio();
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(result['error'] ?? 'Failed to add to portfolio')),
          );
        }
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Invalid input: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Portfolio Manager'),
        backgroundColor: Colors.blue.shade700,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // Search Bar
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.grey.shade100,
            child: TextField(
              controller: _searchController,
              decoration: InputDecoration(
                hintText: 'Search stocks by name or symbol...',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                filled: true,
                fillColor: Colors.white,
                suffixIcon: _isSearching
                    ? const Padding(
                        padding: EdgeInsets.all(12.0),
                        child: SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      )
                    : IconButton(
                        icon: const Icon(Icons.search),
                        onPressed: () => _searchStocks(_searchController.text),
                      ),
              ),
              onChanged: (value) {
                if (value.isNotEmpty) {
                  _searchStocks(value);
                } else {
                  setState(() {
                    _searchResults = [];
                  });
                }
              },
              onSubmitted: _searchStocks,
            ),
          ),

          // Search Results or Portfolio
          Expanded(
            child: _searchController.text.isNotEmpty && _searchResults.isNotEmpty
                ? _buildSearchResults()
                : _buildPortfolio(),
          ),
        ],
      ),
    );
  }

  Widget _buildSearchResults() {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _searchResults.length,
      itemBuilder: (context, index) {
        final stock = _searchResults[index];
        final symbol = stock['Symbol']?.toString() ?? 'N/A';
        final name = stock['Name']?.toString() ?? 'N/A';
        final price = stock['Current_Price']?.toString() ?? 'N/A';
        final sector = stock['Sector']?.toString() ?? 'N/A';

        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          child: ListTile(
            title: Text(
              symbol,
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(name),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Chip(
                      label: Text(sector, style: const TextStyle(fontSize: 12)),
                      backgroundColor: Colors.blue.shade100,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '₹$price',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.green.shade700,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            trailing: ElevatedButton(
              onPressed: () {
                final priceValue = double.tryParse(price) ?? 0.0;
                _addToPortfolio(symbol, priceValue);
              },
              child: const Text('Add'),
            ),
          ),
        );
      },
    );
  }

  Widget _buildPortfolio() {
    if (_isLoadingPortfolio) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_error != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline, size: 64, color: Colors.red.shade300),
            const SizedBox(height: 16),
            Text(_error!, style: const TextStyle(color: Colors.red)),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _loadPortfolio,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    if (_portfolio.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.account_balance, size: 64, color: Colors.grey.shade400),
            const SizedBox(height: 16),
            const Text(
              'Your portfolio is empty',
              style: TextStyle(fontSize: 18, color: Colors.grey),
            ),
            const SizedBox(height: 8),
            const Text(
              'Search for stocks above to add them to your portfolio',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey),
            ),
          ],
        ),
      );
    }

    final totalInvested = _portfolioAnalysis?['total_invested'] ?? 0.0;
    final currentValue = _portfolioAnalysis?['current_value'] ?? 0.0;
    final totalPnl = _portfolioAnalysis?['total_pnl'] ?? 0.0;
    final totalPnlPercent = _portfolioAnalysis?['total_pnl_percent'] ?? 0.0;

    return RefreshIndicator(
      onRefresh: _loadPortfolio,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Portfolio Summary Card
          Card(
            color: Colors.blue.shade50,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Portfolio Summary',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  _buildSummaryRow('Total Invested', '₹${totalInvested.toStringAsFixed(2)}'),
                  _buildSummaryRow('Current Value', '₹${currentValue.toStringAsFixed(2)}'),
                  _buildSummaryRow(
                    'Total P&L',
                    '₹${totalPnl.toStringAsFixed(2)} (${totalPnlPercent.toStringAsFixed(2)}%)',
                    color: totalPnl >= 0 ? Colors.green : Colors.red,
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // Holdings List
          const Text(
            'Holdings',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),

          ..._portfolio.map((holding) => _buildHoldingCard(holding)),
        ],
      ),
    );
  }

  Widget _buildSummaryRow(String label, String value, {Color? color}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 16)),
          Text(
            value,
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHoldingCard(Map<String, dynamic> holding) {
    final symbol = holding['symbol']?.toString() ?? 'N/A';
    final shares = holding['shares']?.toString() ?? '0';
    final buyPrice = holding['buy_price']?.toString() ?? '0';
    final currentPrice = holding['current_price']?.toString() ?? '0';
    final pnl = holding['pnl']?.toString() ?? '0';
    final pnlPercent = holding['pnl_percent']?.toString() ?? '0';
    final pnlValue = double.tryParse(pnl) ?? 0.0;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  symbol,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: pnlValue >= 0 ? Colors.green.shade100 : Colors.red.shade100,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${pnlValue >= 0 ? '+' : ''}${pnlPercent}%',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: pnlValue >= 0 ? Colors.green.shade700 : Colors.red.shade700,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _buildHoldingMetric('Shares', shares),
                _buildHoldingMetric('Buy Price', '₹$buyPrice'),
                _buildHoldingMetric('Current', '₹$currentPrice'),
                _buildHoldingMetric('P&L', '₹$pnl', color: pnlValue >= 0 ? Colors.green : Colors.red),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHoldingMetric(String label, String value, {Color? color}) {
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
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }
}

