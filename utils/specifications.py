"""
This module returns the co-safe ltl specifications that we used in our experiments.
The set of propositional symbols are {a,b,c,d,e,f,g,h,n,s}:
    a: got_wood
    b: used_toolshed
    c: used_workbench
    d: got_grass
    e: used_factory
    f: got_iron
    g: used_bridge
    h: used_axe
    n: is_night
    s: at_shelter
--------------
The set of test symbols are {c,j,k,m,w,x}:
    test:               train:
    c: wood             a
    j: iron             d
    k: workbench        f
    m: toolshed         l
    w: axe              n
    x: grass            q
    
    Train
    get gem -> ((dUa);f);l;w
    make shears -> (aUd);f
    q!;q;(fVl)
    (f!Ul!);l
    ((a;q)V(d;w));f;l!
    From Andreas test
    get gem -> ((j;k) U c); m ; w
    make shears -> (cUj); k
    x!;x;(kVm)
    (k!U m!);m
    ((c;x)V(j;w));k;m!
"""
def get_Complex_specs():
    specifications = []
    # Test
    population='cjkmwx'

    specifications.append('((j;k)Uc);m;w')
    specifications.append('(cUj);k')
    specifications.append('x!;x;(kVm)')
    specifications.append('k!;m!;m')
    specifications.append('((c;x)V(j;w));k;m!')
    
    return specifications, population

def get_specs_ConVLay():
    specifications = []

    # All afirm
    population = 'abcdefghijklmnopqrstuvwxyz' #0-25
    # A handfull of symbols
    # population = 'bcejlprtwy' #0-9
    specifications.append('z')#0
    for e in population:
        specifications.append(e)

    return specifications, population

def get_TTL_specs(trainSz=20, test=False):
    AffSpecs = []
    NegSpecs = []
    DisjSpecs = []

    """
    Below we have the different variants for train and test
    """
    if test:
        # General test objects
        population = 'cjkmwx'
        # population = 'adflnq'
        # population = 'abdflnpqru' #0-9
        # population = 'abdefghilnopqrstuvyz' #0-19
    else:
        # 6-obj train
        if trainSz == 6:
            population = 'adflnq' #0-5
        # 10-obj train
        elif trainSz == 10:
            population = 'abdflnpqru' #0-9
        # 15-obj train
        elif trainSz == 15:
            population = 'abdfghilnopqrtu' #0-14
        #20 obj train 
        else:
            population = 'abdefghilnopqrstuvyz' #0-19
   
    for e in population:
        AffSpecs.append(e)
    for e in population: #6-11
        # specifications.append('!'+e) InvNeg
        NegSpecs.append(e+'!')
    for e in population:
        for j in population:
            if e != j:
                DisjSpecs.append(e+'V'+j)
    # DisjSpecs = ['jVc']
    # DisjSpecs = ['cVj']
    # DisjSpecs = ['jVk']
    # population = 'kj'

    # trained
    # NegSpecs = ['f!']
    # AffSpecs = 'q'

    # Deceptive
    # NegSpecs = ['q!']
    # AffSpecs = 'f'

    # trained with positive only
    # NegSpecs = ['c!']
    # AffSpecs = 'j'

    # test
    # population = 'mx'
    # NegSpecs = ['m!']
    # AffSpecs = ['x']
    # DisjSpecs = ['xVc']
    # DisjSpecs = ['cVx']

    # population = 'an'
    # NegSpecs = ['n!']
    # AffSpecs = ['a']
    # population = 'andq'
    # DisjSpecs = ['aVc']
    # DisjSpecs = ['cVa']
    # DisjSpecs = ['aVn']

     #----- Deceptive
    # NegSpecs = ['a!']
    # AffSpecs = ['n']
    # DisjSpecs = ['nVc']
    # DisjSpecs = ['cVn']
    # DisjSpecs = ['qVd']

    #----- Deceptive
    # NegSpecs = ['x!']
    # AffSpecs = 'm'
    # DisjSpecs = ['mVc']
    # DisjSpecs = ['cVm']


    return AffSpecs, NegSpecs, DisjSpecs, population

def get_learningRL_specs(trainSz=20):
    AffSpecs = []
    NegSpecs = []
    DisjSpecs = []

    """
    Below we have the different variants for train and test
    """
    # 6-obj train
    if trainSz == 6:
        population = 'ejklru' #0-5
    # 10-obj train
    elif trainSz == 10:
        population = 'ehjklmprux' #0-9
    #20 obj train 
    else:
        population = 'acdefghijklmopqrtuxz' #0-19
    # # General test objects
    # population = 'bnsvwy'
    for e in population:
        AffSpecs.append(e)
    for e in population: #6-11
        # specifications.append('!'+e) InvNeg
        NegSpecs.append(e+'!')
    for e in population:
        for j in population:
            if e != j:
                DisjSpecs.append(e+'V'+j)
    # DisjSpecs = ['bVy']
    # population = 'sb'
    # NegSpecs = ['s!']
    # AffSpecs = 'b'
     # ---- Fake ones
    # AffSpecs = 's'
    # NegSpecs = ['b!']

    # DisjSpecs = ['cVy']
    # population = 'cz'
    # NegSpecs = ['c!']
    # AffSpecs = 'z'
    # ---- Fake ones
    # AffSpecs = 's'
    # NegSpecs = ['b!']
    # ----Likelihood
    # population = 'vb'
    # NegSpecs = ['v!']
    # AffSpecs = 'b'

    return AffSpecs, NegSpecs, DisjSpecs, population


def get_Until_specs(trainSz=20, test=False):
    specs = []

    """
    Below we have the different variants for train and test
    """
    if test:
        # General test objects
        population = 'cjkmwx'
        # population = 'adflnq'
    else:
        # 6-obj train
        if trainSz == 6:
            population = 'adflnq' #0-5
        # 10-obj train
        elif trainSz == 10:
            population = 'abdflnpqru' #0-9
        # 15-obj train
        elif trainSz == 15:
            population = 'abdfghilnopqrtu' #0-14
        #20 obj train 
        else:
            population = 'abdefghilnopqrstuvyz' #0-19

    for e in population:
        for j in population:
            if e != j: specs.append(e+'U'+j)

    return specs, population

def get_TTL2_specs(trainSz=20, test=False, mode=0):
    """
    Modes:
    0: standard training specficiations
    1: standard test specifications
    2: Slow movement specifications only
    3: Normal movement specifications only
    4: Fast movement specifications only
    """
    TrueUntilgoal = [] # specs of the type +pU+p'
    FalseUntilgoal = [] # specs of the type -pU+p'
    MoveUntilEscape = [] # specs of type type-of-move U-p 
    PDisjSpecs = []
    NDisjSpecs = []

    # ExtraMove = []

    if test:
        # General test objects
        # population = 'cjkmwx'
        # population = 'abdefghilnopqrstuvyzcjkmwx0123456789'
        # population = 'bcjmptwx0345689' #10
        # population = 'adefghiklnoqrsu' #this one
        population = 'abdfghiflnprtvyz2456'
        # population = 'abdfghiflnprtvyz2456'

        # train
        # population = 'yokzseq1'
        # test
        # population = 'CFHILSKZ'
                     # 'yokzseq1'

        # Not visual
        # population = 'adgiflnprtvz248' #training
        # population = 'cejkmoqsuwx0137' #testing 
        # population =  'BCDFGHIJKLMNOPQRSWYZ' #BCOPQ


    else:
        # 6-obj train
        if trainSz == 6:
            population = 'adflnq' #0-5
        # 10-obj train
        elif trainSz == 10:
            population = 'abdflnpqru' #0-9
        # 15-obj train
        elif trainSz == 15:
            population = 'abdfghilnopqrtu' #0-14
        #20 obj train 
        else:
            # population = 'abdefghilnopqrstuvyz' #0-20
            # population = 'adefghiklnoqrsuvyz12'
            # Not visual
            population = 'abdfghiflnprtvyz2456'

    for e in population:
        # TrueUntilgoal.append('EE   U+' +e) #"E" is the symbol used to express that anything is accepted
        MoveUntilEscape.append('EE   U-' +e) 
        for j in population:
            if e != j:
                FalseUntilgoal.append('-' + e + '   U+'+j) 
                TrueUntilgoal.append('+' + e + '   U+'+j) 
                PDisjSpecs.append('EE   U+' + e +'V+'+j)
                for i in population:
                    if i != e and i!=j:
                        NDisjSpecs.append('-' + e + '   U+'+j + 'V+' + i) 
                        PDisjSpecs.append('+' + e + 'V+'+ i + 'U+'+j)
                        PDisjSpecs.append('+' + e + '   U+'+j +'V+'+ i)
                        for t in population:
                            if t != e and t!=j and t!= i:
                                PDisjSpecs.append('+' + e + 'V+'+ j + 'U+'+ i \
                                    + 'V+'+t)
    # print('Comprueba TrueUnitl Goal',NDisjSpecs)
    # jajsaj+=1

    return TrueUntilgoal, FalseUntilgoal, MoveUntilEscape, PDisjSpecs,\
        NDisjSpecs, population
def get_MTTL2_specs(trainSz=20, test=False, mode=0):
    """
    Modes:
    0: standard training specficiations
    1: standard test specifications
    2: Slow movement specifications only
    3: Normal movement specifications only
    4: Fast movement specifications only
    """
    TrueUntilgoal = [] # specs of the type +pU+p'
    FalseUntilgoal = [] # specs of the type -pU+p'
    MoveUntilEscape = [] # specs of type type-of-move U-p 

    ExtraMove = []


    """
    Below we have the different variants for train and test
    """
    # mode=4

    if mode==1: mv=['F']
    elif mode==2: mv=['S']
    elif mode==3: mv=['N']
    elif mode==4: mv=['F']
    else:
        mode=0
        mv=['S', 'N']


    if test:
        # General test objects
        population = 'cjkmwx'
        population = 'abdefghilnopqrstuvyz' #0-20
        # population = 'adflnq'
    else:
        # 6-obj train
        if trainSz == 6:
            population = 'adflnq' #0-5
        # 10-obj train
        elif trainSz == 10:
            population = 'abdflnpqru' #0-9
        # 15-obj train
        elif trainSz == 15:
            population = 'abdfghilnopqrtu' #0-14
        #20 obj train 
        else:
            population = 'abdefghilnopqrstuvyz' #0-20

    for e in population:
        for move in mv:
            TrueUntilgoal.append(move + 'EEU+' +e) #"E" is the symbol used to express that anything is accepted
            MoveUntilEscape.append(move + 'EEU-' +e) 
        for j in population:
            if e != j:
                for move in mv:
                    FalseUntilgoal.append(move + '-' + e + 'U+'+j) 
                    if mode != 1:
                        TrueUntilgoal.append(move + '+' + e + 'U+'+j) 
                if mode == 0:
                    ExtraMove.append('F+' + e + 'U+'+j) 
    # if test:
    #     # General test objects
    #     population = 'cjkmwx'
    #     # population = 'adflnq'
    #     for e in population:
    #         Motion_Reach_Specs.extend(['FEEUP'+ j, 'FEEU!'+ j])
    #         for j in population:
    #             if e != j: 
    #                 Safe_Reach_Specs.extend(['FP'+ e +'U'+'P'+ j,
    #                                         'F!'+ e +'U'+'P'+j,])
    # else:
    #     # 6-obj train
    #     if trainSz == 6:
    #         population = ' adflnq' #0-5
    #     # 10-obj train
    #     elif trainSz == 10:
    #         population = ' abdflnpqru' #0-9
    #     # 15-obj train
    #     elif trainSz == 15:
    #         population = ' abdfghilnopqrtu' #0-14
    #     #20 obj train 
    #     else:
    #         population = ' abdefghilnopqrstuvyz' #0-20

    #     for e in population:
    #         Motion_Reach_Specs.extend(['SEEUP'+ j, 'SEEU!'+ j,
    #                                 'NEEUP'+ j, 'NEEU!'+ j])
    #         for j in population:
    #             if e != j: 
    #                 Safe_Reach_Specs.extend(['SP'+ e +'U'+'P'+ j, 'S!'+ e +'U'+'P'+j,
    #                                     'NP'+ e +'U'+'P'+j, 'N!'+ e +'U'+'P'+j,
    #                                     'FP'+ e +'U'+'P'+ j])
    return TrueUntilgoal, FalseUntilgoal, MoveUntilEscape, ExtraMove, population