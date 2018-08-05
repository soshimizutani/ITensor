
//
// Distributed under the ITensor Library License, Version 1.2
//    (See accompanying LICENSE file.)
//
#ifndef __ITENSOR_DMRG_H
#define __ITENSOR_DMRG_H

#include "itensor/eigensolver.h"
#include "itensor/mps/localmposet.h"
#include "itensor/mps/localmpo_mps.h"
#include "itensor/mps/sweeps.h"
#include "itensor/mps/DMRGObserver.h"
#include "itensor/util/cputime.h"

#include "../../soshi_dmrg/myfunctions/write_to_file.cpp"
#include "itensor/mps/mpo.h"

using std::vector;


namespace itensor {

    //
    // Available DMRG methods:
    //

    //
    //DMRG with an MPO
    //
    template <class Tensor>
    Real
    dmrg(MPSt<Tensor>& psi,
         const MPOt<Tensor>& H,
         const Sweeps& sweeps,
         const Args& args = Global::args())
    {
        LocalMPO<Tensor> PH(H,args);
        Real energy = DMRGWorker(psi,PH,sweeps,args);
        return energy;
    }

    //
    //DMRG with an MPO and custom DMRGObserver
    //
    template <class Tensor>
    Real
    dmrg(MPSt<Tensor>& psi,
         const MPOt<Tensor>& H,
         const Sweeps& sweeps,
         DMRGObserver<Tensor>& obs,
         const Args& args = Global::args())
    {
        LocalMPO<Tensor> PH(H,args);
        Real energy = DMRGWorker(psi,PH,sweeps,obs,args);
        return energy;
    }

    //
    //DMRG with an MPO and boundary tensors LH, RH
    // LH - H1 - H2 - ... - HN - RH
    //(ok if one or both of LH, RH default constructed)
    //
    template <class Tensor>
    Real
    dmrg(MPSt<Tensor>& psi,
         const MPOt<Tensor>& H,
         const Tensor& LH, const Tensor& RH,
         const Sweeps& sweeps,
         const Args& args = Global::args())
    {
        LocalMPO<Tensor> PH(H,LH,RH,args);
        Real energy = DMRGWorker(psi,PH,sweeps,args);
        return energy;
    }

    //
    //DMRG with an MPO and boundary tensors LH, RH
    //and a custom observer
    //
    template <class Tensor>
    Real
    dmrg(MPSt<Tensor>& psi,
         const MPOt<Tensor>& H,
         const Tensor& LH, const Tensor& RH,
         const Sweeps& sweeps,
         DMRGObserver<Tensor>& obs,
         const Args& args = Global::args())
    {
        LocalMPO<Tensor> PH(H,LH,RH,args);
        Real energy = DMRGWorker(psi,PH,sweeps,obs,args);
        return energy;
    }

    //
    //DMRG with a set of MPOs (lazily summed)
    //(H vector is 0-indexed)
    //
    template <class Tensor>
    Real
    dmrg(MPSt<Tensor>& psi,
         const std::vector<MPOt<Tensor> >& Hset,
         const Sweeps& sweeps,
         const Args& args = Global::args())
    {
        LocalMPOSet<Tensor> PH(Hset,args);
        Real energy = DMRGWorker(psi,PH,sweeps,args);
        return energy;
    }

    //
    //DMRG with a set of MPOs and a custom DMRGObserver
    //(H vector is 0-indexed)
    //
    template <class Tensor>
    Real
    dmrg(MPSt<Tensor>& psi,
         const std::vector<MPOt<Tensor> >& Hset,
         const Sweeps& sweeps,
         DMRGObserver<Tensor>& obs,
         const Args& args = Global::args())
    {
        LocalMPOSet<Tensor> PH(Hset,args);
        Real energy = DMRGWorker(psi,PH,sweeps,obs,args);
        return energy;
    }

    //
    //DMRG with a single Hamiltonian MPO and a set of
    //MPS to orthogonalize against
    //(psis vector is 0-indexed)
    //Named Args recognized:
    // Weight - real number w > 0; calling dmrg(psi,H,psis,sweeps,Args("Weight",w))
    //          sets the effective Hamiltonian to be
    //          H + w * (|0><0| + |1><1| + ...) where |0> = psis[0], |1> = psis[1]
    //          etc.
    //
    template <class Tensor>
    Real
    dmrg(MPSt<Tensor>& psi,
         const MPOt<Tensor>& H,
         const std::vector<MPSt<Tensor> >& psis,
         const Sweeps& sweeps,
         const Args& args = Global::args())
    {
        LocalMPO_MPS<Tensor> PH(H,psis,args);
        Real energy = DMRGWorker(psi,PH,sweeps,args);
        return energy;
    }

    //
    //DMRG with a single Hamiltonian MPO,
    //a set of MPS to orthogonalize against,
    //and a custom DMRGObserver.
    //(psis vector is 0-indexed)
    //Named Args recognized:
    // Weight - real number w > 0; calling dmrg(psi,H,psis,sweeps,Args("Weight",w))
    //          sets the effective Hamiltonian to be
    //          H + w * (|0><0| + |1><1| + ...) where |0> = psis[0], |1> = psis[1]
    //          etc.
    //
    template <class Tensor>
    Real
    dmrg(MPSt<Tensor>& psi,
         const MPOt<Tensor>& H,
         const std::vector<MPSt<Tensor> >& psis,
         const Sweeps& sweeps,
         DMRGObserver<Tensor>& obs,
         const Args& args = Global::args())
    {
        LocalMPO_MPS<Tensor> PH(H,psis,args);
        Real energy = DMRGWorker(psi,PH,sweeps,obs,args);
        return energy;
    }



    //
    // DMRGWorker
    //

    template <class Tensor, class LocalOpT>
    Real inline
    DMRGWorker(MPSt<Tensor>& psi,
               LocalOpT& PH,
               const Sweeps& sweeps,
               const Args& args = Global::args())
    {
        DMRGObserver<Tensor> obs(psi,args);
        Real energy = DMRGWorker(psi,PH,sweeps,obs,args);
        return energy;
    }

    template <class Tensor, class LocalOpT>
    Real
    DMRGWorker(MPSt<Tensor>& psi,
               LocalOpT& PH,
               const Sweeps& sweeps,
               DMRGObserver<Tensor>& obs,
               Args args = Global::args())
    {
        const bool quiet = args.getBool("Quiet",false);
        const int debug_level = args.getInt("DebugLevel",(quiet ? 0 : 1));

        const int N = psi.N();
        Real energy = NAN;

        psi.position(1);

        args.add("DebugLevel",debug_level);
        args.add("DoNormalize",true);
        println("===========================================================================");
        println("-----------------------------DMRG sweep------------------------------------");
        println("===========================================================================");
        printf(" |%-5s|  %-8s|   %-8s | %-4s |   %-10s |  %-6s | %-6s |", "Sweep", "Energy", "S_ent", "max m", "max trun", "CPU(s)", "Wall(s)");

        std::vector<double> E_v, SvN_v, bond_v, trun_v, cpu_v, wall_v;

        auto save_psi=Args::global().getBool("DMRG_save_psi");


        auto runname=Args::global().getString("runname");
        std::string filename="psi0";
        
        if(save_psi){
          write_to_file::wavefunction(psi, filename);
        }

        for(int sw = 1; sw <= sweeps.nsweep(); ++sw)
        {


            cpu_time sw_time;
            args.add("Sweep",sw);
            args.add("Cutoff",sweeps.cutoff(sw));
            args.add("Minm",sweeps.minm(sw));
            args.add("Maxm",sweeps.maxm(sw));
            args.add("Noise",sweeps.noise(sw));
            args.add("MaxIter",sweeps.niter(sw));

            if(!PH.doWrite()
               && args.defined("WriteM")
               && sweeps.maxm(sw) >= args.getInt("WriteM"))
            {
                if(!quiet)
                {
                    println("\nTurning on write to disk, write_dir = ",
                            args.getString("WriteDir","./"));
                }

                //psi.doWrite(true);
                PH.doWrite(true);
            }

            for(int b = 1, ha = 1; ha <= 2; sweepnext(b,ha,N))
            {
                if(!quiet)
                {
                    printfln("Sweep=%d, HS=%d, Bond=(%d,%d)",sw,ha,b,(b+1));
                }

                PH.position(b,psi);

                auto phi = psi.A(b)*psi.A(b+1);

                energy = davidson(PH,phi,args);

                auto spec = psi.svdBond(b,phi,(ha==1?Fromleft:Fromright),PH,args);


                if(!quiet)
                {
                    printfln("    Truncated to Cutoff=%.1E, Min_m=%d, Max_m=%d",
                             sweeps.cutoff(sw),
                             sweeps.minm(sw),
                             sweeps.maxm(sw) );
                    printfln("    Trunc. err=%.1E, States kept: %s",
                             spec.truncerr(),
                             showm(linkInd(psi,b)) );
                }

                obs.lastSpectrum(spec);

                args.add("AtBond",b);
                args.add("HalfSweep",ha);
                args.add("Energy",energy);
                args.add("Truncerr",spec.truncerr());

                obs.measure(args);

            } //for loop over b

            auto sm = sw_time.sincemark();
            //printfln("    Sweep %d CPU time = %s (Wall time = %s)",
            //         sw,showtime(sm.time),showtime(sm.wall));

            E_v.push_back(energy);
            SvN_v.push_back(args.getReal("S_ent"));
            bond_v.push_back(args.getInt("Largest m"));
            trun_v.push_back(args.getReal("Largest truncation"));
            cpu_v.push_back(sm.time);
            wall_v.push_back(sm.wall);


            //save psi to a file at each DMRG iteration
            if(save_psi){
            filename="psi"+std::to_string(sw);
            write_to_file::wavefunction(psi, filename);
            }

            printf(" | %-3d | %-8g | %-10g |  %-4d | %-5e | %-6s | %-6s |", sw, energy, args.getReal("S_ent"), args.getInt("Largest m"), args.getReal("Largest truncation"),showtime(sm.time), showtime(sm.wall));


            if(obs.checkDone(args) ) break;

        } //for loop over sw

        filename ="E.dat";
        write_to_file::vector(E_v, filename);

        filename = "SvN.dat";
        write_to_file::vector(SvN_v, filename);

        filename = "bond.dat";
        write_to_file::vector(bond_v, filename);

        filename = "trun.dat";
        write_to_file::vector(trun_v, filename);

        filename = "bond.dat";
        write_to_file::vector(bond_v, filename);

        filename = "cpu.dat";
        write_to_file::vector(cpu_v, filename);

        filename = "wall.dat";
        write_to_file::vector(wall_v, filename);






        println();
        println("===========================================================================");
        println("===========================================================================");

        psi.normalize();

        return energy;
    }

} //namespace itensor


#endif
