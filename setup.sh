mkdir -p ~/.streamlit/
echo "
[general]n
email = "vatsalpatel2035@gmail.com"n
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truen
enableCORS=falsen
port = $PORTn
" > ~/.streamlit/config.toml
