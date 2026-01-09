import 'package:flutter/material.dart';
import 'screens/portfolio_screen.dart';
import 'screens/chart_analysis_screen.dart';
import 'widgets/chat_overlay.dart';

void main() {
  runApp(const FinSamaritanApp());
}

class FinSamaritanApp extends StatelessWidget {
  const FinSamaritanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FinSamaritan - Portfolio Manager with AI Agent',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
      ),
      home: const MainScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;
  bool _showChatOverlay = false;

  final List<Widget> _screens = [
    const PortfolioScreen(),
    const ChartAnalysisScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Main content
          IndexedStack(
            index: _currentIndex,
            children: _screens,
          ),
          
          // Chat Overlay
          if (_showChatOverlay)
            ChatOverlay(
              onClose: () {
                setState(() {
                  _showChatOverlay = false;
                });
              },
            ),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.account_balance),
            label: 'Portfolio',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.show_chart),
            label: 'Chart Analysis',
          ),
        ],
        type: BottomNavigationBarType.fixed,
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          setState(() {
            _showChatOverlay = true;
          });
        },
        child: const Icon(Icons.chat),
        tooltip: 'AI Assistant',
      ),
    );
  }
}

