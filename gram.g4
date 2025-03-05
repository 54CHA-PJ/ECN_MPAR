grammar gram;

program
    : defstates defactions transitions EOF
    ;

defstates
    : STATES statedef (',' statedef)* ';'
    ;

statedef
    : ID (':' INT)?
    ;

defactions
    : ACTIONS ID (',' ID)* ';'
    ;

transitions
    : trans+
    ;

trans
    : transact
    | transnoact
    ;

transact
    : ID '[' ID ']' FLECHE INT ':' ID ('+' INT ':' ID)* ';'
    ;

transnoact
    : ID FLECHE INT ':' ID ('+' INT ':' ID)* ';'
    ;

STATES  : 'States';
ACTIONS : 'Actions';
DPOINT  : ':' ;
FLECHE  : '->';
SEMI    : ';';
VIRG    : ',';
PLUS    : '+';
LCROCH  : '[';
RCROCH  : ']';

INT : [0-9]+ ;
ID  : [a-zA-Z_][a-zA-Z0-9_]* ;
WS  : [ \t\n\r\f]+ -> skip ;
