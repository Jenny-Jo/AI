#!/usr/bin/env python
# -*- coding: EUC-KR -*-

import cgi

html_body = """
<html>
<body>
<p>����� �̸��� <span style="font-size:48px"> %s </span> �Դϴ�!</p>
</body>
</html>
"""

form = cgi.FieldStorage()

print("Content-type: text/html")
print(html_body % form['name'].value)