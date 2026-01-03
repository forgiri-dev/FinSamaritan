// Implementation for mobile and desktop platforms (where dart:io is available)
import 'dart:io';

String getPlatformType() {
  if (Platform.isAndroid) {
    return 'android';
  } else if (Platform.isIOS) {
    return 'ios';
  } else if (Platform.isWindows) {
    return 'windows';
  } else if (Platform.isLinux) {
    return 'linux';
  } else if (Platform.isMacOS) {
    return 'macos';
  }
  return 'unknown';
}

