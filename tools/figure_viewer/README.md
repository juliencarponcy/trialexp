For deployment, please see [link](https://shiny.posit.co/py/get-started/deploy-on-prem.html)

Key files:
1. App folders should be in `/srv/shiny-server`
2. Configuration file is in `/etc/shiny-server/shiny-server.conf`
3. Use `sudo systemctl restart shiny-server` to apply new change
4. app is available at port `3838`
