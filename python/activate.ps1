# Ativa o ambiente virtual e mostra informações úteis

Write-Host "🐍 Ativando ambiente virtual Python..." -ForegroundColor Cyan

# Ativar venv
& ".\.venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "✅ Ambiente virtual ativado!" -ForegroundColor Green
Write-Host ""
Write-Host "📦 Versão do Python:" -ForegroundColor Yellow
python --version
Write-Host ""
Write-Host "🔧 Comandos úteis:" -ForegroundColor Yellow
Write-Host "  ruff check .          - Verificar código"
Write-Host "  ruff check . --fix    - Corrigir automaticamente"
Write-Host "  ruff format .         - Formatar código"
Write-Host "  python train.py       - Treinar modelo"
Write-Host ""
Write-Host "📚 Para mais informações, veja README.md" -ForegroundColor Cyan
Write-Host ""
