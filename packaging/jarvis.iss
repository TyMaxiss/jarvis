; Установщик Джарвиса (Inno Setup 6). Собирается после PyInstaller (в CI — задача «Сборка exe»):
;   iscc /DAppVersion=0.6.0 /DSourceDir=<папка dist\Jarvis> /DOutputDir=<куда> /DIconFile=<.ico> packaging\jarvis.iss
; Ставит для текущего пользователя, без прав администратора, в %LOCALAPPDATA%\Programs\Jarvis:
; программа хранит рядом с собой настройки, модели и логи, поэтому её папка должна быть доступна
; для записи. Тихое обновление из программы (assistant/updates.py):
;   Jarvis-Setup-<версия>.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="<папка>" /RELAUNCH=yes

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\Jarvis"
#endif
#ifndef OutputDir
  #define OutputDir "..\setup"
#endif
; Версия файла — только цифры: 0.6.0.dev0 → 0.6.0.
#if Pos(".dev", AppVersion) > 0
  #define FileVersion Copy(AppVersion, 1, Pos(".dev", AppVersion) - 1)
#else
  #define FileVersion AppVersion
#endif

[Setup]
; AppId не меняется никогда: по нему новая версия ставится поверх старой.
AppId={{6F1C2B84-5A7E-4E2B-9C1D-3A8F0E6B7D21}
AppName=Джарвис
AppVersion={#AppVersion}
AppVerName=Джарвис {#AppVersion}
AppPublisher=TyMaxiss
AppPublisherURL=https://github.com/TyMaxiss/jarvis
AppSupportURL=https://github.com/TyMaxiss/jarvis/issues
AppUpdatesURL=https://github.com/TyMaxiss/jarvis/releases
VersionInfoVersion={#FileVersion}
VersionInfoProductName=Джарвис
DefaultDirName={autopf}\Jarvis
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=Jarvis-Setup-{#AppVersion}
#ifdef IconFile
SetupIconFile={#IconFile}
#endif
UninstallDisplayIcon={app}\Jarvis.exe
UninstallDisplayName=Джарвис
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
; Программу закрывает не Restart Manager, а сама программа (обновление) или пользователь — см. [Code].
CloseApplications=no
RestartApplications=no
ShowLanguageDialog=no

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Обновление: библиотеки прежней версии не должны остаться рядом с новыми.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Джарвис"; Filename: "{app}\Jarvis.exe"
Name: "{autodesktop}\Джарвис"; Filename: "{app}\Jarvis.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Jarvis.exe"; Description: "{cm:LaunchProgram,Джарвис}"; Flags: nowait postinstall skipifsilent
; После обновления из программы — так же, как с ярлыка: через Проводник, с его окружением и без
; установщика среди родителей (у запущенной установщиком копии голос не запускался). Без оболочки
; Windows (Проводник не запущен) — напрямую.
Filename: "{win}\explorer.exe"; Parameters: """{app}\Jarvis.exe"""; Flags: nowait; Check: RelaunchViaShell
Filename: "{app}\Jarvis.exe"; Flags: nowait; Check: RelaunchDirectly

[UninstallDelete]
; Ярлык автозапуска создаёт сама программа (настройка «Запускать вместе с Windows»).
Type: files; Name: "{userstartup}\Джарвис.lnk"

[Code]
const
  AppMutex = 'Local\JarvisVoiceAssistant';
  CloseRequest = 'Джарвис запущен. Закройте его («Выход» в меню значка в трее) и нажмите OK.';

{ Тихое обновление из программы запускает её заново: /RELAUNCH=yes. }
function Relaunch(): Boolean;
begin
  Result := WizardSilent() and (ExpandConstant('{param:RELAUNCH|no}') = 'yes');
end;

{ Оболочка Windows запущена: есть панель задач Проводника. }
function ShellRunning(): Boolean;
begin
  Result := FindWindowByClassName('Shell_TrayWnd') <> 0;
end;

function RelaunchViaShell(): Boolean;
begin
  Result := Relaunch() and ShellRunning();
end;

function RelaunchDirectly(): Boolean;
begin
  Result := Relaunch() and not ShellRunning();
end;

{ Через Проводник переменную окружения не передать: что программу запустили после обновления,
  она узнаёт по отметке в своей папке (assistant/frozen.py, UPDATED_MARK) и убирает её. }
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and Relaunch() then
    SaveStringToFile(ExpandConstant('{app}\updated.mark'), GetDateTimeString('yyyy-mm-dd hh:nn:ss', '-', ':'), False);
end;

{ Файл открыт работающей программой: exe, который выполняется, нельзя открыть на запись. }
function FileInUse(const Path: String): Boolean;
var
  Stream: TFileStream;
begin
  Result := False;
  if not FileExists(Path) then
    Exit;
  try
    Stream := TFileStream.Create(Path, fmOpenReadWrite or fmShareExclusive);
    Stream.Free;
  except
    Result := True;
  end;
end;

{ Джарвис работает: занят его мьютекс, а когда известна папка программы (CheckFiles) — ещё
  и её exe. Мьютекс освобождается чуть раньше, чем процесс завершается, а JarvisConsole.exe
  бывает занят процессом голоса Silero. }
function AppRunning(CheckFiles: Boolean): Boolean;
begin
  Result := CheckForMutexes(AppMutex);
  if not Result and CheckFiles then
    Result := FileInUse(ExpandConstant('{app}\Jarvis.exe')) or
              FileInUse(ExpandConstant('{app}\JarvisConsole.exe'));
end;

function WaitForAppExit(Seconds: Integer; CheckFiles: Boolean): Boolean;
var
  I: Integer;
begin
  for I := 1 to Seconds * 4 do
  begin
    if not AppRunning(CheckFiles) then
      Break;
    Sleep(250);
  end;
  Result := not AppRunning(CheckFiles);
end;

function AskToClose(CheckFiles: Boolean): Boolean;
begin
  Result := True;
  while AppRunning(CheckFiles) do
  begin
    if MsgBox(CloseRequest, mbInformation, MB_OKCANCEL) = IDCANCEL then
    begin
      Result := False;
      Exit;
    end;
    WaitForAppExit(10, CheckFiles);  { программа закрывается не мгновенно }
  end;
end;

{ Программа запускает обновление и сразу закрывается: установщик ждёт, пока она выйдет.
  Папка программы здесь ещё не известна, поэтому проверяется только мьютекс. }
function InitializeSetup(): Boolean;
begin
  if WizardSilent() then
    Result := WaitForAppExit(60, False)
  else
    Result := AskToClose(False);
end;

{ Перед копированием файлов: exe прежней версии должны освободиться. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not WaitForAppExit(30, True) then
    Result := 'Джарвис ещё работает. Закройте его и запустите установку снова.';
end;

function InitializeUninstall(): Boolean;
begin
  if UninstallSilent() then
    Result := WaitForAppExit(60, True)
  else
    Result := AskToClose(True);
end;

{ Настройки, ключи, свои команды и скачанные модели лежат в папке программы: удалить их — по желанию. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Folder: String;
begin
  Folder := ExpandConstant('{app}');
  if (CurUninstallStep = usPostUninstall) and DirExists(Folder) and not UninstallSilent() then
    if MsgBox('Удалить также настройки, ключи API, свои команды, скачанные модели и библиотеки?' + #13#10 +
              'Они лежат в папке ' + Folder + '.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      DelTree(Folder, True, True, True);
end;
