#define MyAppName "ai2npu"
#define MyAppVersion "0.1.15"
#define MyAppPublisher "ai2npu"
#define DistDir "..\dist\ai2npu"

[Setup]
AppId=ai2npu
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\ai2npu
DefaultGroupName=ai2npu
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
OutputDir=..\dist
OutputBaseFilename=ai2npu-setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes

[Dirs]
Name: "{commonappdata}\ai2npu"; Permissions: users-modify
Name: "{commonappdata}\ai2npu\logs"; Permissions: users-modify
Name: "{commonappdata}\ai2npu\cache"; Permissions: users-modify
Name: "{commonappdata}\ai2npu\models"; Permissions: users-readexec

[Tasks]
Name: "bge"; Description: "Install BAAI/bge-m3 embedding model"; GroupDescription: "Models:"
Name: "whisper"; Description: "Install OpenVINO whisper-large-v3-turbo-int8-ov model"; GroupDescription: "Models:"

[Files]
Source: "{#DistDir}\ai2npu.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#DistDir}\*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#DistDir}\config.example.toml"; DestDir: "{app}"; Flags: ignoreversion

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
; Requires Microsoft Visual C++ Redistributable 2015-2022 x64 on target machines
; because ai2npu_genai_bridge.dll links to the official MSVC OpenVINO GenAI runtime.
Filename: "{app}\ai2npu.exe"; Parameters: "install-model --model ""BAAI/bge-m3"" --model-dir ""{commonappdata}\ai2npu\models\strokinkv\bge-m3-int8-ov"""; Tasks: bge; Flags: runhidden waituntilterminated
Filename: "{app}\ai2npu.exe"; Parameters: "install-model --model ""openai/whisper-large-v3-turbo"" --model-dir ""{commonappdata}\ai2npu\models\OpenVINO\whisper-large-v3-turbo-int8-ov"""; Tasks: whisper; Flags: runhidden waituntilterminated
Filename: "{app}\ai2npu.exe"; Parameters: "init-config --path ""{commonappdata}\ai2npu\config.toml"" --data-dir ""{commonappdata}\ai2npu"""; Flags: runhidden waituntilterminated
Filename: "{app}\ai2npu.exe"; Parameters: "install-service --config ""{commonappdata}\ai2npu\config.toml"" --exe ""{app}\ai2npu.exe"""; Check: ServiceMissing; Flags: runhidden waituntilterminated
Filename: "{app}\ai2npu.exe"; Parameters: "start-service"; Flags: runhidden waituntilterminated

[UninstallRun]
Filename: "{app}\ai2npu.exe"; Parameters: "stop-service"; Flags: runhidden waituntilterminated; RunOnceId: "StopAi2NpuService"
Filename: "{app}\ai2npu.exe"; Parameters: "uninstall-service"; Flags: runhidden waituntilterminated; RunOnceId: "RemoveAi2NpuService"

[Code]
procedure StopExistingService;
var
  ResultCode: Integer;
begin
  Exec(ExpandConstant('{sys}\sc.exe'), 'stop ai2npuService', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(3000);

  { Last resort for upgrades: kill only the process owned by this service. }
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /FI "SERVICES eq ai2npuService"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(1000);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  StopExistingService;
  Result := '';
end;

function ServiceMissing: Boolean;
var
  ResultCode: Integer;
begin
  Exec(ExpandConstant('{sys}\sc.exe'), 'query ai2npuService', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := ResultCode <> 0;
end;

function NeedsAddPath(Path: string): Boolean;
var
  CurrentPath: string;
begin
  if not RegQueryStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', CurrentPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(CurrentPath) + ';') = 0;
end;

procedure WriteDefaultConfig;
var
  ConfigPath: string;
  DataDir: string;
  ModelRoot: string;
  Text: string;
begin
  ConfigPath := ExpandConstant('{commonappdata}\ai2npu\config.toml');
  if FileExists(ConfigPath) then
  begin
    exit;
  end;

  DataDir := ExpandConstant('{commonappdata}\ai2npu');
  ModelRoot := DataDir + '\models';
  Text :=
    '[server]' + #13#10 +
    'host = "127.0.0.1"' + #13#10 +
    'port = 9555' + #13#10 +
    'request_body_limit_mb = 100' + #13#10 +
    'thread_count = 16' + #13#10 + #13#10 +
    '[queue]' + #13#10 +
    'max_pending_requests = 10' + #13#10 +
    'default_timeout_sec = 600' + #13#10 + #13#10 +
    '[logging]' + #13#10 +
    'level = "info"' + #13#10 +
    'directory = ''' + DataDir + '\logs''' + #13#10 +
    'max_file_size_mb = 10' + #13#10 +
    'max_files = 10' + #13#10;

  if WizardIsTaskSelected('bge') then
  begin
    Text := Text + #13#10 +
    '[[models]]' + #13#10 +
    'id = "BAAI/bge-m3"' + #13#10 +
    'type = "embedding"' + #13#10 +
    'path = ''' + ModelRoot + '\strokinkv\bge-m3-int8-ov''' + #13#10 +
    'enabled = true' + #13#10 +
    'preload = false' + #13#10 +
    'idle_timeout_sec = 0' + #13#10 +
    'queue_timeout_sec = 600' + #13#10 +
    'normalize = true' + #13#10;
  end;

  if WizardIsTaskSelected('whisper') then
  begin
    Text := Text + #13#10 +
    '[[models]]' + #13#10 +
    'id = "openai/whisper-large-v3-turbo"' + #13#10 +
    'type = "whisper"' + #13#10 +
    'path = ''' + ModelRoot + '\OpenVINO\whisper-large-v3-turbo-int8-ov''' + #13#10 +
    'enabled = true' + #13#10 +
    'preload = false' + #13#10 +
    'idle_timeout_sec = 0' + #13#10 +
    'queue_timeout_sec = 600' + #13#10 +
    'max_audio_duration_sec = 1800' + #13#10;
  end;

  SaveStringToFile(ConfigPath, Text, False);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    WriteDefaultConfig;
  end;
end;
