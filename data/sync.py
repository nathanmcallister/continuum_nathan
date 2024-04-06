#!/bin/python3
import pysftp
import os
import shutil

puppet_info_filename = "../.puppet.hidden"
file = open(puppet_info_filename, "r")
puppet_info = file.readline().strip().split("@")
user = puppet_info[0]
host = puppet_info[1]

# Setup .sftp_temp directory and make sure it is empty
if not os.path.isdir(".sftp_temp"):
    os.mkdir(".sftp_temp")
else:
    shutil.rmtree(".sftp_temp")
    os.mkdir(".sftp_temp")

# Connect vis SFTP
connection = pysftp.Connection(
    host=host, username=user
)

# Get regs and tip_cals
connection.chdir("github/tracker-serial-interface")
connection.get_r("regs", ".sftp_temp")
connection.get_r("tip_cals", ".sftp_temp")

connection.close()

shutil.copytree(".sftp_temp/regs", "regs", dirs_exist_ok=True)
shutil.copytree(".sftp_temp/tip_cals", "tip_cals", dirs_exist_ok=True)

shutil.rmtree(".sftp_temp")

