# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# Cleanup colors
# '@' means "wall"
# 'H' is potential waste spawn point
# 'R' is river cell
# 'S' is stream cell
# 'P' means "player" spawn point
# 'A' means apple spawn point
# 'B' is potential apple spawn point
# ' ' is empty space

# CLEANUP LIO
CLEANUP_LIO3_MAP = [
    '@@@@@@@@@@',
    '@HH   P B@',
    '@RR    BB@',
    '@HH     B@',
    '@RR    BB@',
    '@HH P   B@',
    '@RR    BB@',
    '@HH     B@',
    '@RRP   BB@',
    '@@@@@@@@@@']

CLEANUP_LIO2_MAP = [
    '@@@@@@@',
    '@H  PB@',
    '@H   B@',
    '@    B@',
    '@    B@',
    '@ P  B@',
    '@@@@@@@']

# n=2
HARVEST_EASY_MAP = [
    '@@@@@@@@@@@@@@@@@@@@',
    '@P AAAAAAAAAAAAAA P@',
    '@  AAAAAAAAAAAAAA  @',
    '@@@@@@@@@@@@@@@@@@@@'
]

CLEANUP_EASY_MAP = [
    '@@@@@',
    '@BPH@',
    '@BPH@',
    '@@@@@'
]

# n=12
HARVEST_MEDIUM_MAP = [
    '@@@@@@@@@@@@@@@@@@@@',
    '@PP AAAAAAAAAAAA PP@',
    '@PP AAAAAAAAAAAA PP@',
    '@PP AAAAAAAAAAAA PP@',
    '@@@@@@@@@@@@@@@@@@@@'
]
 # n=9
CLEANUP_MEDIUM_MAP = [
    '@@@@@@@@@@@@@@@@@@@@',
    '@BBBBB PPP HHHHRRRR@',
    '@BBBBB PPP HHHHRRRR@',
    '@BBBBB PPP HHHHRRRR@',
    '@@@@@@@@@@@@@@@@@@@@'
]

# n=3
HARVEST_LITTLE_MAP = [
    '@@@@@@@@@@@@@@@@@@@@',
    '@ P       P       P@',
    '@     AA AA  AA    @',
    '@    AAA AA AAA    @',
    '@@@@@@@@@@@@@@@@@@@@',
]

# n=5
HARVEST_PART_MAP = [
    '@@@@@@@@@@@@@@@@@@@@',
    '@ P       P       P@',
    '@     AA AA  AA    @',
    '@    AAA AA AAA    @',
    '@    AAA A    A    @',
    '@     A AAA  AA    @',
    '@       A A AAA    @',
    '@  P        P      @',
    '@@@@@@@@@@@@@@@@@@@@',]

# n=10
HARVEST_HALF_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P           P          P    P  @',
    '@        A   AA         AAA    A     @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@    AAA A    A  A AAA  A  A   A A   @',
    '@    A A       AAA A  AAA            @',
    '@      AAA  AAA  A      AAA   AAA    @',
    '@   P      P          P      P   P   @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',]

# n=20
HARVEST_WALL_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P@ AAAAA    P  A P  @',
    '@  P     A P AA    @P    AAA    A  A  @',
    '@     A AAA  AAA   @ A    A AA AAAA   @',
    '@ A  AAA A    A  A @AAA  A  A   A A   @',
    '@AAA  A A    A  AAA@ A  AAA P      A P@',
    '@ A A  AAA  AAA  A @A    A AA   AA AA @',
    '@  A A  AAA    A A @ AAA    AAA  A    @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@   AAA  A      AAA@  A    AAAA       @',
    '@ P  A  P    A  A A@AA    A  A      P @',
    '@A  AAA  A  A  AAA @A    AAAA     P   @',
    '@    A A  PAAA PA A@   A  A AA   A  A @',
    '@     AAA   A A  AA@A  A   AA   AAA P @',
    '@ A    A     AAA  A@  A          A    @',
    '@       P     A    @     A  P A     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

HARVEST_FIVE_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@            A      AAAAA       A    @',
    '@        A P AA         AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A        @',
    '@A  AAA  A  A  AAA A    AAAA         @',
    '@    A A   AAA  A A      A AA   A    @',
    '@     AAA   A A  AAA      AA   AAA   @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A                      @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

# n=5
CLEANUP_FIVE_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH    P BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHH P      BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR     P BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH P     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@',]

# n=10
CLEANUP_TEN_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH    P BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHH P      BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR     P BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH P     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH    P BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHH P      BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR     P BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH P     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@',
]

# n=20
HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

# n=10
CLEANUP_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH      BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR  P    BBBB@',
    '@RRRRR    P BBBBB@',
    '@HHHHH       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR   P P BBBB@',
    '@HHHHH   P  BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH P   BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH    P  BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH  P P BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@']
