
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; MODULE      : init-sympy.scm
;; DESCRIPTION : Initialize the Test Py plugin
;; COPYRIGHT   : (C) 2019  Darcy Shen
;;
;; This software falls under the GNU general public license version 3 or later.
;; It comes WITHOUT ANY WARRANTY WHATSOEVER. For details, see the file LICENSE
;; in the root directory or <http://www.gnu.org/licenses/gpl-3.0.html>.
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (test_py-serialize lan t)
    (with u (pre-serialize lan t)
      (with s (texmacs->code (stree->tree u) "SourceCode")
        (string-append s "\n<EOF>\n"))))

(define (test_py-launcher)
 ; (string-append (python-command) " \""
 ;                 "/home/ingolf/.TeXmacs/plugins/test_py/progs/tm_test_py.py\"")
;   (display python-command)
 ; )
  (if (url-exists? "$TEXMACS_HOME_PATH/plugins/test_py")
      (string-append (python-command) " \""
                     (getenv "TEXMACS_HOME_PATH")
                     "/plugins/test_py/progs/tm_test_py.py\"")
;      (string-append (python-command) " \""
;                     (getenv "TEXMACS_HOME_PATH")
;                     "/plugins/test_py/tm_test_py.py\"")))
      (display "Falsch abgebogen")
      ))

(plugin-configure test_py
  (:require (python-command))
  (:launch ,(test_py-launcher))
  (:serializer ,test_py-serialize)
  (:tab-completion #t)
  (:session "Test Py"))
