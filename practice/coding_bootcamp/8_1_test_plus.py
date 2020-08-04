import unittest

#test 대상 plus 함수
def plus(a,b):
    return a+b

class PlusTest(unittest.TestCase): # class이름에  Test넣기, unittest module의 TestCase Class 상속할 것
    #test programme
    def test_plus(self): # 'test_'넣고 테스트 함수 이르 설정
        self.assertEqual(10,plus(2,8))
        self.assertEqual(20,plus(2,8))

if __name__ =="__main__":
    unittest.main()
'''    
F
======================================================================
FAIL: test_plus (__main__.PlusTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "f:\Study\practice\coding_bootcamp\8_1_test_plus.py", line 11, in test_plus
    self.assertEqual(20,plus(2,8))
AssertionError: 20 != 10

---------------------------------------'-'------------------------------
Ran 1 test in 0.000s

FAILED (failures=1)'''