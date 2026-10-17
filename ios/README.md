# Scan iPad app

Open `Scan.xcodeproj` in Xcode with an iOS 26 or newer SDK. To regenerate the
project after adding source files, install XcodeGen and run `bash ios/generate_project.sh`
from the repository root. Select your own development team for device installation.

## Reliability behavior

| Operation | Expected behavior |
| --- | --- |
| Local recording | Does not create a TCP sender or wait for workstation configuration. |
| Port entry | Accepts 1–65535; invalid input is not saved. Previously saved invalid ports recover to 7001. |
| Check workstation | Refreshes both TCP reachability and the model list, with visible loading and failure states. |
| Live start / upload | Requires a successful acknowledgement of the selected VLM before sending frames. Automatic launch uses the same gate. |
| Stream disconnect | Stops after a prolonged stall and reports an unconfirmed interruption; it does not claim receipt or mapping completion. |
| Damaged recording | Lists the problem separately and preserves the files. The user can move the folder into `recordings-recovery` and export it using Files. Valid pending recordings remain available. |
| Record detail | Refreshes status, instances, objects and processing timing every three seconds while processing; preserves the selected view. Leaving the page/tab cancels polling. Refresh retries the whole detail view. |

## Validation

Run the production Foundation/Combine code tests on macOS, without a workstation
or camera:

```sh
swift test --package-path ios
```

The tests cover offline start, invalid ports, failed/recovered model discovery,
host changes, model acknowledgement failures, cancelled starts, disconnect
classification, damaged-record recovery with byte preservation, and detail refresh
through processing completion. GitHub Actions runs the same suite.

Compile the complete app without signing or installing it:

```sh
xcodebuild -project ios/Scan.xcodeproj -scheme Scan -configuration Debug \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
```

Physical-device acceptance remains separate: verify a local scan with the
workstation offline, a live scan with model acknowledgement, network interruption,
and a record progressing to completion. Unit tests and compilation do not validate
ARKit camera/depth capture or the availability of a remote model runtime.
