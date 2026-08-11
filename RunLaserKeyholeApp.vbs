' Run the Laser Keyhole CFD GUI without showing a console window
Set WshShell = CreateObject("WScript.Shell")
Dim repoDir
repoDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\") - 1)
WshShell.Run "cmd /c cd /d """ & repoDir & """ && pythonw """ & repoDir & "\keyhole-cfd\app\laserkeyhole_app.pyw"""", 0, False
Set WshShell = Nothing
