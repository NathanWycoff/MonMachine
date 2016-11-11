library(RSelenium)

startServer(dir='/bin/', args = c("-webdriver.gecko.driver=/home/nathan/Downloads/gecko/geckodriver"))

remDir <- remoteDriver$new()

#Sys.setenv(webdriver.gecko.driver="/home/nathan/Downloads/gecko/geckodriver");


remDir$open()

