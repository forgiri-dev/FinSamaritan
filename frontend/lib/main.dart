import 'package:flutter/material.dart';
import 'screens/screener_screen.dart';
import 'screens/chart_analysis_screen.dart';
import 'screens/compare_screen.dart';

void main() {
  runApp(const FinSamaritanApp());
}

class FinSamaritanApp extends StatelessWidget {
  const FinSamaritanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FinSamaritan - Agentic AI Financial Assistant',
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

  final List<Widget> _screens = [
    const ScreenerScreen(),
    const ChartAnalysisScreen(),
    const CompareScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.search),
            label: 'Screener',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.show_chart),
            label: 'Chart Doctor',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.compare_arrows),
            label: 'Compare',
          ),
        ],
        type: BottomNavigationBarType.fixed,
      ),
    );
  }
}

