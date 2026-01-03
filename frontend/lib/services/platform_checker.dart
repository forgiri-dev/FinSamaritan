// Platform checker with conditional imports
// This file uses conditional imports to avoid Platform errors on web

import 'platform_checker_stub.dart'
    if (dart.library.io) 'platform_checker_io.dart'
    as platform_checker;

String getPlatformType() => platform_checker.getPlatformType();

