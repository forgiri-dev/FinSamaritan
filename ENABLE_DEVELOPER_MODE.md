# How to Enable Developer Mode on Windows

## Quick Method (Recommended)

1. **Open Windows Settings:**
   - Press `Windows Key + I` to open Settings
   - OR run this command in PowerShell/Command Prompt:
     ```
     start ms-settings:developers
     ```

2. **Navigate to Developer Mode:**
   - Click on **"Privacy & security"** in the left sidebar
   - Click on **"For developers"** (or search for "Developer" in settings)
   - OR directly go to: Settings → Update & Security → For developers

3. **Enable Developer Mode:**
   - Find the toggle switch for **"Developer Mode"**
   - Turn it **ON**
   - Windows may ask for confirmation - click **"Yes"**
   - You may need to restart your computer (Windows will prompt you)

4. **Verify it's enabled:**
   - The toggle should show as **ON**
   - You should see a message saying "Developer Mode is on"

## Alternative: Enable via PowerShell (Run as Administrator)

If the Settings app doesn't work, you can enable it via PowerShell:

```powershell
# Run PowerShell as Administrator
# Right-click PowerShell → "Run as Administrator"

# Enable Developer Mode
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" -Name "AllowDevelopmentWithoutDevLicense" -Value 1
```

Then restart your computer.

## After Enabling Developer Mode

1. **Restart your computer** (if Windows prompted you)
2. **Try building again:**
   ```powershell
   cd frontend
   flutter clean
   flutter pub get
   flutter run -d windows
   ```

## If Developer Mode Still Doesn't Work

### Option 1: Use Flutter Web Instead
```powershell
cd frontend
flutter run -d chrome
```

### Option 2: Check Flutter Doctor
```powershell
flutter doctor -v
```
Make sure Windows desktop support is enabled and Visual Studio is properly configured.

### Option 3: Use Android Emulator
If you have Android Studio installed:
```powershell
cd frontend
flutter run -d <android-emulator-id>
```

## Troubleshooting

**If you get "Access Denied" errors:**
- Make sure you're running PowerShell/Command Prompt as Administrator
- Check Windows User Account Control (UAC) settings

**If Developer Mode toggle is grayed out:**
- You may need to be logged in as an Administrator account
- Some Windows editions (like Windows Home) may have limited developer features

**If symlinks still don't work:**
- Try running Flutter from an Administrator terminal
- Check if your antivirus is blocking symlink creation

