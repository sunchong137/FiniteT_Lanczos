# Copyright 2016-2023 Chong Sun (sunchong137@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
from datetime import datetime

stdout = sys.stdout

def time():
    stdout.write(datetime.now().strftime("%y %b %d %H:%M:%S") + "  ")
    stdout.flush()

def result(msg, *args):
    time()
    stdout.write("********" + "  " + msg + "\n")
    stdout.flush()

def section(msg, *args):
    time()
    stdout.write("########" + "  " + msg + "\n")
    stdout.flush()

if __name__ == "__main__":
    msg = "This is a test"
    result(msg)
    section(msg)
           
