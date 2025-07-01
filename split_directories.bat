for /f "delims=" %%A in ('dir /a:d /b /s "Mayo *"') do (
    if "%%~nxA"=="Mayo 0" (
        xcopy /y "%%~A" "2class\Mayo 0\"
    ) else if "%%~nxA"=="Mayo 1" (
        xcopy /y "%%~A" "2class\Mayo 1+\"
    ) else if "%%~nxA"=="Mayo 2" (
        xcopy /y "%%~A" "2class\Mayo 1+\"
    ) else if "%%~nxA"=="Mayo 3" (
        xcopy /y "%%~A" "2class\Mayo 1+\"
    )
)