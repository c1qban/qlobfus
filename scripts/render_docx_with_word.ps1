param(
    [Parameter(Mandatory = $true)]
    [string]$InputDocx,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [string]$PopplerBin = ""
)

$ErrorActionPreference = "Stop"
$docxPath = (Resolve-Path -LiteralPath $InputDocx).Path
$outDir = New-Item -ItemType Directory -Force -Path $OutputDir
$pdfPath = Join-Path $outDir.FullName (([System.IO.Path]::GetFileNameWithoutExtension($docxPath)) + ".pdf")

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$document = $null
try {
    $document = $word.Documents.Open($docxPath, $false, $true)
    $document.ExportAsFixedFormat($pdfPath, 17)
}
finally {
    if ($document -ne $null) {
        $document.Close($false)
    }
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}

if ($PopplerBin) {
    $env:Path = "$PopplerBin;$env:Path"
}

$prefix = Join-Path $outDir.FullName "page"
pdftoppm -png -r 120 $pdfPath $prefix | Out-Null

Get-ChildItem -Path $outDir.FullName -Filter "page-*.png" |
    Sort-Object Name |
    Select-Object FullName, Length
