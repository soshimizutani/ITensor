#include "itensor/all.h"

using namespace itensor;

int main()
    {
    //
    // Single-site wavefunction
    //
    
    //Make a dimension 2 Index
    Index s = Index("s",2);

    //Construct an ITensor
    ITensor psi = ITensor(s); //default initialized to zero

    //Set first element to 1 (initialize the spin to be up)
    psi.set(s(1),1);
        Print(psi);
        //        (out) psi =
        //        ITensor r=1: ("s",2,Link|78)
        //        {log(scale)=0.00, norm=1.00 (Dense Real)}

        PrintData(psi);
//        (out)psi =
//        ITensor r=1: ("s",2,Link|78)
//        {log(scale)=0.00, norm=1.00 (Dense Real)}
//        (1) 1.0000000

    
    // Operators
    ITensor Sz = ITensor(s,prime(s));
    ITensor Sx = ITensor(s,prime(s));

    // indexing from 1, not 0
    Sz.set(s(1),prime(s)(1),+0.5);
    Sz.set(s(2),prime(s)(2),-0.5);

    Sx.set(s(1),prime(s)(2),+0.5);
    Sx.set(s(2),prime(s)(1),+0.5);

    PrintData(Sz);
//    (out)Sz =
//    ITensor r=2: ("s",2,Link|78) ("s",2,Link|78)'
//    {log(scale)=0.00, norm=0.71 (Dense Real)}
//    (1,1) 0.5000000
//    (2,2) -0.500000
        
        // Note:  The last number(|78) is part of the id number of the Index. Id numbers are random 64 bit integers and vary each time you run your program.
   
    PrintData(Sx);
//    (out)Sx =
//    ITensor r=2: ("s",2,Link|78) ("s",2,Link|78)'
//    {log(scale)=0.00, norm=0.71 (Dense Real)}
//    (2,1) 0.5000000
//    (1,2) 0.5000000

    
    // Product Sx * phi
    ITensor phi = Sx * psi;
    
    PrintData(phi);
//    (out)phi =
//    ITensor r=1: ("s",2,Link|78)'
//    {log(scale)=-0.69, norm=0.50 (Dense Real)}
//    (2) 0.5000000
        
    
    phi.noprime();

    PrintData(phi);
        
        
    // <Sx>
        ITensor cpsi=dag(prime(psi));//<psi|
    // Note: cpsi=prime(psi) would give the same result( dag() is not necessary if psi is all real)
        
    PrintData(cpsi);
//    (out)cpsi =
//    ITensor r=1: ("s",2,Link|159)'
//    {log(scale)=0.00, norm=1.00 (Dense Real)}
//    (1) 1.0000000

    
    ITensor zz=(cpsi * Sz * psi);
    PrintData(zz);
//    (out)zz =
//    ITensor r=0:
//    {log(scale)=-0.69, norm=0.50 (Dense Real)}
//    0.5000000 
        
    println("<Sz> =", zz.real());
//      (out) <Sz> = 0.5
        

        
        

 
//    Real theta = Pi/4;
//
//    //Extra factors of two come from S=1/2 representation
//    psi.set(s(1),cos(theta/2));
//    psi.set(s(2),sin(theta/2));
//
//    PrintData(psi);
//
//    // Expectation values
//    auto cpsi = dag(prime(psi));
//    PrintData(cpsi);
//
//    Real zz = (cpsi * Sz * psi).real();
//    Real xx = (cpsi * Sx * psi).real();
//
//    println("<Sz> = ", zz);
//    println("<Sx> = ", xx);
//
//    Real zzz =overlap(psi, Sz, psi);
        
    return 0;
    }
