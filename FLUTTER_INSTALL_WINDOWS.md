# Flutter Installation Guide for Windows

## Quick Fix: "flutter is not recognized"

If you see this error: **"flutter is not recognized as the name of a cmdlet, function, script file, or operable program"**, Flutter is not installed or not in your PATH.

---

## Method 1: Install Flutter SDK (Recommended)

### Step 1: Download Flutter

1. Visit: https://docs.flutter.dev/get-started/install/windows
2. Download the latest **stable** Flutter SDK ZIP file
3. The file is large (~1.5 GB), so it may take a few minutes

### Step 2: Extract Flutter

1. Create a folder: `C:\src` (if it doesn't exist)
2. Extract the ZIP file to `C:\src\flutter`
   - **Important:** Extract directly to `C:\src\flutter` (not `C:\src\flutter\flutter`)
   - After extraction, you should have: `C:\src\flutter\bin\flutter.bat`

### Step 3: Add Flutter to PATH

**Option A: Using GUI (Easier)**

1. Press `Windows Key + X` → Select **"System"**
2. Click **"Advanced system settings"** (on the right)
3. Click **"Environment Variables"** button (at the bottom)
4. Under **"User variables"**, find **"Path"** and click **"Edit"**
5. Click **"New"**
6. Add: `C:\src\flutter\bin`
7. Click **"OK"** on all dialogs

**Option B: Using PowerShell (Advanced)**

Open PowerShell as Administrator and run:

```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\src\flutter\bin", "User")
```

### Step 4: Verify Installation

1. **Close ALL PowerShell/Command Prompt windows**
2. Open a **NEW** PowerShell window
3. Run:
   ```powershell
   flutter --version
   ```

You should see something like:
```
Flutter 3.x.x • channel stable • ...
```

### Step 5: Run Flutter Doctor

```powershell
flutter doctor
```

This checks your setup. You'll see what's installed and what's missing.

---

## Method 2: Quick Test (Temporary)

If you just want to test without permanent installation:

```powershell
# Replace C:\src\flutter with your actual Flutter path
$env:Path += ";C:\src\flutter\bin"
flutter --version
```

**Note:** This only works in the current PowerShell session. After closing, you'll need to add to PATH permanently.

---

## Troubleshooting

### "flutter is still not recognized"

1. **Did you close all terminal windows?** (Required for PATH changes)
2. **Is the path correct?** Check: `C:\src\flutter\bin\flutter.bat` exists
3. **Try restarting your computer**
4. **Check PATH manually:**
   ```powershell
   $env:Path -split ';' | Select-String flutter
   ```
   Should show your Flutter bin path

### "Flutter doctor shows errors"

This is normal! You don't need everything. For this project:

**Minimum required:**
- ✅ Flutter SDK installed
- ✅ Chrome (for web) OR Android Studio (for Android) OR Visual Studio (for Windows desktop)

**Common setups:**

**For Web Development (Chrome):**
- Just install Flutter SDK
- Chrome should already be installed
- Run: `flutter run -d chrome`

**For Android Development:**
- Install Android Studio: https://developer.android.com/studio
- Open Android Studio → SDK Manager → Install Android SDK
- Run: `flutter doctor --android-licenses` (accept all)
- Run: `flutter run` (select Android device/emulator)

**For Windows Desktop:**
- Install Visual Studio 2022: https://visualstudio.microsoft.com/
- Install "Desktop development with C++" workload
- Run: `flutter run -d windows`

---

## Alternative: Use Git (For Developers)

If you have Git installed:

```powershell
cd C:\src
git clone https://github.com/flutter/flutter.git -b stable
# Then add C:\src\flutter\bin to PATH (same as Step 3 above)
```

---

## Verify Installation

After installation, test in a NEW PowerShell window:

```powershell
# Check Flutter version
flutter --version

# Check setup
flutter doctor

# Check available devices
flutter devices
```

---

## Next Steps

Once Flutter is installed:

1. Navigate to frontend directory:
   ```powershell
   cd C:\Users\124sa\FinSamaritan\frontend
   ```

2. Install dependencies:
   ```powershell
   flutter pub get
   ```

3. Run the app:
   ```powershell
   flutter run -d chrome
   ```

---

## Still Having Issues?

1. **Check Flutter installation path:**
   - Should be: `C:\src\flutter\bin\flutter.bat`
   - If different, update PATH with your actual path

2. **Try running with full path:**
   ```powershell
   C:\src\flutter\bin\flutter.bat --version
   ```

3. **Check Windows Defender/Antivirus:**
   - Some antivirus software blocks Flutter
   - Add Flutter folder to exceptions if needed

4. **Re-download Flutter:**
   - Delete old installation
   - Download fresh ZIP from official site
   - Extract and add to PATH again

5. **Check for multiple Flutter installations:**
   - Search for `flutter.bat` on your system
   - Make sure PATH points to the correct one

---

## Useful Commands

```powershell
# Check Flutter version
flutter --version

# Check setup status
flutter doctor

# Check available devices
flutter devices

# Update Flutter
flutter upgrade

# Clean build cache (if issues)
flutter clean
```

