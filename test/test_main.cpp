#include <iostream>
#include <fstream>

#include "lattice/lattice_base.h"
#include "lattice/square.h"
#include "lattice/cubic.h"

#include "model/model_base.h"
#include "model/repulsive_hubbard.h"
#include "model/attractive_hubbard.h"

#include "measure/observable.h"
#include "measure/observable_handler.h"
#include "measure/measure_handler.h"

#include "dqmc_walker.h"
#include "dqmc.h"
#include "dqmc_initializer.h"
#include "dqmc_io.h"

#include "svd_stack.h"
#include "utils/linear_algebra.hpp"
#include "utils/numerical_stable.hpp"
#include "random.h"

#include "checkerboard/checkerboard_base.h"
#include "checkerboard/square.h"

#include <chrono>

#include <unistd.h>
#include "utils/progressbar.hpp"

#include "utils/mpi.hpp"

#include "utils/toml.hpp"

// #include "random.h"
// #include "hubbard.h"

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>


#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/local_time/local_time.hpp>


int main(int argc, char* argv[]) {


    
    
    // test cubic lattice

    Lattice::Cubic* cubic = new Lattice::Cubic();

    cubic->set_lattice_params({10,10,10});
    cubic->initial();

    // std::cout << cubic->Index2Momentum( cubic->GammaPointIndex() ) << std::endl;
    // std::cout << cubic->Index2Momentum( cubic->XPointIndex() ) << std::endl;
    // std::cout << cubic->Index2Momentum( cubic->MPointIndex() ) << std::endl;
    // std::cout << cubic->Index2Momentum( cubic->RPointIndex() ) << std::endl;
    // std::cout << cubic->Index2Momentum( cubic->DeltaLineIndex() ) << std::endl;

    // for ( auto k : cubic->LambdaLineIndex() ) {
    //     std::cout << cubic->Index2Momentum( k ) << "\n" << std::endl;
    // }

    // for ( auto k : cubic->SLineIndex() ) {
    //     std::cout << cubic->Index2Momentum( k ) << "\n" << std::endl;
    // }

    // for ( auto k : cubic->TLineIndex() ) {
    //     std::cout << cubic->Index2Momentum( k ) << "\n" << std::endl;
    // }

    // std::cout << cubic->NearestNeighbour(0,0) << std::endl;
    // std::cout << cubic->NearestNeighbour(0,1) << std::endl;
    // std::cout << cubic->NearestNeighbour(0,2) << std::endl;
    // std::cout << cubic->NearestNeighbour(0,3) << std::endl;
    // std::cout << cubic->NearestNeighbour(0,4) << std::endl;
    // std::cout << cubic->NearestNeighbour(0,5) << std::endl;

    std::cout << cubic->Index2Site(337) << std::endl;
    std::cout << cubic->Index2Site(123) << std::endl;
    std::cout << cubic->Index2Site( cubic->Displacement(123,337) ) << std::endl;



    // // number of k stars for cubic lattice
    // for (int ll = 2; ll <= 20; ll++) {
    //     int kstar_num = 0;
    //     for ( int i = 1; i <= std::floor(ll/2.0)+1; i++ ) {
    //         kstar_num += i*(i+1)/2;
    //     }

    //     const auto halfl = std::floor(ll/2.0)+1;
    //     std::cout << halfl*(halfl+1)*(2*halfl+1)/12 + halfl*(halfl+1)/4 << "   "
    //               << kstar_num << std::endl;

    //     // std::cout << (std::floor(ll/2.0)+1)*(std::floor(ll/2.0)+2)/2 
    //     //           << std::endl;
    // }

    


    // std::cout << cubic->m_index2site_table << std::endl;
    // std::cout << cubic->m_index2site_table.size() << std::endl;

    // std::cout << cubic->m_space_size << std::endl;
    // std::cout << cubic->m_hopping_matrix << std::endl;
    // std::cout << cubic->m_hopping_matrix.size() << std::endl;
    
















    // boost::mpi::environment env(argc, argv);
    // boost::mpi::communicator world;
    // const int master = 0;
    // const int rank = world.rank();

    // // -------------------------------------------------------------------------------------------
    // //                             Output current date and time
    // // -------------------------------------------------------------------------------------------
    // if ( rank == master ) {
    //     const auto current_time = boost::posix_time::second_clock::local_time();
    //     std::cout << boost::format(" Current time: %s \n") % current_time << std::endl;
    // }


    // // -------------------------------------------------------------------------------------------
    // //                             Output MPI and hardware info
    // // -------------------------------------------------------------------------------------------
    // if ( rank == master ) {
    //     // print MPI and hardware information
    //     boost::format fmt_mpi(" Distribute tasks to %s processes, with the master process being %s. \n");
    //     std::cout << fmt_mpi % world.size() % env.processor_name() << std::endl;
    // }


    // // set up random seeds
    // Utils::Random::set_seed( std::time(nullptr) + rank );
    // // // fixed random seed for debug
    // // Utils::Random::set_seed( 12345 );


    // std::unique_ptr<Model::ModelBase> model;
    // std::unique_ptr<Lattice::LatticeBase> lattice;
    // std::unique_ptr<QuantumMonteCarlo::DqmcWalker> walker;
    // std::unique_ptr<Measure::MeasureHandler> meas_handler;
    // std::unique_ptr<CheckerBoard::CheckerBoardBase> checkerboard;


    // // parse parmas from the configuation file
    // QuantumMonteCarlo::DqmcInitializer::parse_toml_config( 
    //         "../../example/config.toml", world.size(),
    //         model, lattice, walker, meas_handler, checkerboard );

    // // initialize models
    // if ( checkerboard ) { 
    //     // using checkerboard break-up
    //     QuantumMonteCarlo::DqmcInitializer::initial_modules( *model, *lattice, *walker, *meas_handler, *checkerboard ); 
    // }
    // else { 
    //     QuantumMonteCarlo::DqmcInitializer::initial_modules( *model, *lattice, *walker, *meas_handler ); 
    // }

    // // randomly initialize the bosonic fields
    // // todo: read fields from file
    // model->set_bosonic_fields_to_random();
    // if (rank == master ) { 
    //     std::cout << " Configurations of auxiliary fields set to random. \n" << std::endl; 
    // }
    // // QuantumMonteCarlo::DqmcIO::read_bosonic_fields_from_file( "../out.dat", *model);

    // // initialize modules for dqmc
    // QuantumMonteCarlo::DqmcInitializer::initial_dqmc( *model, *lattice, *walker, *meas_handler );

    // if ( rank == master ) {
    //     std::cout << " Initialization finished. \n\n" 
    //               << " The simulation is going to get started with parameters shown below : \n\n"
    //               << std::endl;
    // }

    // // output the initialization info
    // if ( rank == master ) {
    //     QuantumMonteCarlo::DqmcIO::output_init_info 
    //         ( 
    //             std::cout, world.size(), 
    //             *model, *lattice, *walker, *meas_handler, checkerboard 
    //         );
    // }

    // QuantumMonteCarlo::Dqmc::show_progress_bar( (rank == master) );
    // QuantumMonteCarlo::Dqmc::progress_bar_format( 60, '=', ' ' );
    // QuantumMonteCarlo::Dqmc::set_refresh_rate( 10 );
    
    // // start the dqmc simulation
    // QuantumMonteCarlo::Dqmc::timer_begin();
    // QuantumMonteCarlo::Dqmc::thermalize( *walker, *model, *lattice, *meas_handler );
    // QuantumMonteCarlo::Dqmc::measure( *walker, *model, *lattice, *meas_handler );
    
    // // gather observable objects from other processes
    // Utils::MPI::mpi_gather( world, *meas_handler );

    // // perform the analysis
    // QuantumMonteCarlo::Dqmc::analyse( *meas_handler );

    // // end the timer
    // QuantumMonteCarlo::Dqmc::timer_end();

    // // output the ending info
    // if ( rank == master ) {
    //     QuantumMonteCarlo::DqmcIO::output_ending_info( std::cout, *walker );
    // }

    
    // // std::ofstream outfile("../out.dat");
    // // QuantumMonteCarlo::DqmcIO::output_observable_in_bins(
    // //     outfile, meas_handler->find<Observable::VectorObs>("density_of_states")
    // // );
    // // outfile.close();

    // // QuantumMonteCarlo::DqmcIO::output_k_stars( std::cout, *lattice );

    // // QuantumMonteCarlo::DqmcIO::output_imaginary_time_grids( std::cout, *walker );

    // // std::ofstream outfile("../out.dat");
    // // QuantumMonteCarlo::DqmcIO::output_bosonic_fields( outfile, *model );
    // // outfile.close();


    // if ( rank == master ) 
    // {
    //     if ( meas_handler->find("equaltime_sign") ) {
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("equaltime_sign") );
    //     }

    //     if ( meas_handler->find("dynamic_sign") ) {
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("dynamic_sign") );
    //     }

    //     std::cout << std::endl;
      
    //     if ( meas_handler->find("filling_number") ) {
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("filling_number") );
    //     }

    //     if ( meas_handler->find("double_occupancy") ) {
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("double_occupancy") );
    //     }

    //     if ( meas_handler->find("kinetic_energy") ) {
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("kinetic_energy") );
    //     }

    //     if ( meas_handler->find("local_spin_corr") ) {
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("local_spin_corr") );
    //     }

    //     if ( meas_handler->find("momentum_distribution") ) {
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("momentum_distribution") );
    //     }

    //     if ( meas_handler->find("spin_density_structure_factor") ) {
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("spin_density_structure_factor") );
    //     }

    //     if ( meas_handler->find("charge_density_structure_factor") ) {
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("charge_density_structure_factor") );
    //     }

    //     if ( meas_handler->find("s_wave_pairing_corr") ) {
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("s_wave_pairing_corr") );
    //     }

    //     // important!!!
    //     // todo: the correctness of superfluid stiffness measurements should be further checked in attractive hubbard model
    //     if ( meas_handler->find("superfluid_stiffness") ) {  
    //         QuantumMonteCarlo::DqmcIO::output_observable( 
    //             std::cout, meas_handler->find<Observable::ScalarObs>("superfluid_stiffness") );
    //     }
    // }

















    // // minimized version without mpi

    // const auto current_time = boost::posix_time::second_clock::local_time();
    // std::cout << boost::format(" Current time: %s \n") % current_time << std::endl;

    // Utils::Random::set_seed( 12345 );

    // std::unique_ptr<Model::ModelBase> model;
    // std::unique_ptr<Lattice::LatticeBase> lattice;
    // std::unique_ptr<QuantumMonteCarlo::DqmcWalker> walker;
    // std::unique_ptr<Measure::MeasureHandler> meas_handler;
    // std::unique_ptr<CheckerBoard::CheckerBoardBase> checkerboard;


    // // parse parmas from the configuation file
    // QuantumMonteCarlo::DqmcInitializer::parse_toml_config( 
    //         "../config.toml", 1,
    //         model, lattice, walker, meas_handler, checkerboard );

    // // initialize models
    // if ( checkerboard ) { 
    //     // using checkerboard break-up
    //     QuantumMonteCarlo::DqmcInitializer::initial_modules( *model, *lattice, *walker, *meas_handler, *checkerboard ); 
    // }
    // else { 
    //     QuantumMonteCarlo::DqmcInitializer::initial_modules( *model, *lattice, *walker, *meas_handler ); 
    // }

    // // randomly initialize the bosonic fields
    // // todo: read fields from file
    // model->set_bosonic_fields_to_random();
    // std::cout << " Configurations of auxiliary fields set to random. \n" << std::endl; 
    // // QuantumMonteCarlo::DqmcIO::read_bosonic_fields_from_file( "../out.dat", *model);

    // // initialize modules for dqmc
    // QuantumMonteCarlo::DqmcInitializer::initial_dqmc( *model, *lattice, *walker, *meas_handler );

    // std::cout << " Initialization finished. \n\n" 
    //           << " The simulation is going to get started with parameters shown below : \n\n"
    //           << std::endl;

    // // output the initialization info
    // QuantumMonteCarlo::DqmcIO::output_init_info 
    //     ( 
    //         std::cout, 1, 
    //         *model, *lattice, *walker, *meas_handler, checkerboard 
    //     );

    // QuantumMonteCarlo::Dqmc::show_progress_bar( true );
    // QuantumMonteCarlo::Dqmc::progress_bar_format( 60, '=', ' ' );
    
    // // start the dqmc simulation
    // QuantumMonteCarlo::Dqmc::timer_begin();
    // QuantumMonteCarlo::Dqmc::thermalize( *walker, *model, *lattice, *meas_handler );
    // QuantumMonteCarlo::Dqmc::measure( *walker, *model, *lattice, *meas_handler );

    // // perform the analysis
    // QuantumMonteCarlo::Dqmc::analyse( *meas_handler );

    // // end the timer
    // QuantumMonteCarlo::Dqmc::timer_end();

    // // output the ending info
    // QuantumMonteCarlo::DqmcIO::output_ending_info( std::cout, *walker );

    // if ( meas_handler->find("equaltime_sign") ) {
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //         std::cout, meas_handler->find<Observable::ScalarObs>("equaltime_sign") );
    // }

    // if ( meas_handler->find("dynamic_sign") ) {
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //         std::cout, meas_handler->find<Observable::ScalarObs>("dynamic_sign") );
    // }

    // std::cout << std::endl;
  
    // if ( meas_handler->find("filling_number") ) {
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //         std::cout, meas_handler->find<Observable::ScalarObs>("filling_number") );
    // }

    // if ( meas_handler->find("double_occupancy") ) {
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //         std::cout, meas_handler->find<Observable::ScalarObs>("double_occupancy") );
    // }

    // if ( meas_handler->find("kinetic_energy") ) {
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //         std::cout, meas_handler->find<Observable::ScalarObs>("kinetic_energy") );
    // }

    // if ( meas_handler->find("local_spin_corr") ) {
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //         std::cout, meas_handler->find<Observable::ScalarObs>("local_spin_corr") );
    // }

    // if ( meas_handler->find("momentum_distribution") ) {
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //         std::cout, meas_handler->find<Observable::ScalarObs>("momentum_distribution") );
    // }

    // if ( meas_handler->find("spin_density_structure_factor") ) {
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //          std::cout, meas_handler->find<Observable::ScalarObs>("spin_density_structure_factor") );
    // }

    // if ( meas_handler->find("charge_density_structure_factor") ) {
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //         std::cout, meas_handler->find<Observable::ScalarObs>("charge_density_structure_factor") );
    // }

    // if ( meas_handler->find("s_wave_pairing_corr") ) {
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //         std::cout, meas_handler->find<Observable::ScalarObs>("s_wave_pairing_corr") );
    // }

    // // important!!!
    // // todo: the correctness of superfluid stiffness measurements should be further checked in attractive hubbard model
    // if ( meas_handler->find("superfluid_stiffness") ) {  
    //     QuantumMonteCarlo::DqmcIO::output_observable( 
    //         std::cout, meas_handler->find<Observable::ScalarObs>("superfluid_stiffness") );
    // }




























    // // set up params
    // lattice->set_lattice_params({ll,ll});
    // // lattice should be initialized once lattice params have been set.
    // lattice->initial();
    // walker->set_physical_params(beta, lt);
    // walker->set_stabilization_pace(nwrap);
    // model->set_model_params(hopping_t, onsite_u, chemical_potential);
    // meas_handler->set_measure_params(sweeps_warmup, bin_num, bin_size, sweeps_between_bins);
    // meas_handler->set_observables(obs_list);

    // // make sure that the lattice module has been initialized
    // if ( lattice->InitialStatus() ) {
    //     // todo: not necessary ? integrated into the toml-read process
    //     QuantumMonteCarlo::DqmcInitializer::set_measured_momentum(*meas_handler, lattice->MPointIndex());
    //     QuantumMonteCarlo::DqmcInitializer::set_measured_momentum_list(*meas_handler, lattice->kStarsIndex());
    // }

    // // initialize modules
    // // lattice module has been initialized before, so in this function 
    // // it is provided for the initialization of other modules
    // QuantumMonteCarlo::DqmcInitializer::initial_modules(*lattice, *model, *walker, *meas_handler);

    // // using checkerboard break-up
    // // CheckerBoard::CheckerBoardBase* checkerboard = new CheckerBoard::Square();
    // // QuantumMonteCarlo::DqmcInitializer::initial_modules(*lattice, *model, *walker, *meas_handler, *checkerboard);

    // model->set_bosonic_fields_to_random();

    // QuantumMonteCarlo::DqmcInitializer::initial_dqmc(*lattice, *model, *walker, *meas_handler);

    // // walker->sweep_from_0_to_beta(*model);
    // // walker->sweep_from_beta_to_0(*model);
    // // walker->sweep_for_dynamic_greens(*model);
    // // walker->sweep_from_beta_to_0(*model);

    // // QuantumMonteCarlo::Dqmc::sweep_forth_and_back(*walker, *model, *lattice, *meas_handler);

    // // std::chrono::steady_clock::time_point begin_t{}, end_t{};
    // // begin_t = std::chrono::steady_clock::now();


    // QuantumMonteCarlo::Dqmc::show_progress_bar( true );
    // QuantumMonteCarlo::Dqmc::progress_bar_format( 70, '=', ' ' );

    // QuantumMonteCarlo::Dqmc::thermalize(*walker, *model, *lattice, *meas_handler);
    // // std::cout << QuantumMonteCarlo::Dqmc::timer() << std::endl;

    // // int loop = 1e3;
    // // for (int i = 0; i < loop; ++i) {
    // //     walker->wrap_from_0_to_beta(*model, 0);
    // //     walker->wrap_from_beta_to_0(*model, 1);
    // // }


    // // end_t = std::chrono::steady_clock::now();
    // // std::cout << "warm-up : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - begin_t).count() << std::endl;


    // QuantumMonteCarlo::Dqmc::measure(*walker, *model, *lattice, *meas_handler);
    // // std::cout << QuantumMonteCarlo::Dqmc::timer() << std::endl;
    // QuantumMonteCarlo::Dqmc::analyse(*meas_handler);

    // // if (meas_handler->find("filling_number")) {
    // //     auto obs = meas_handler->find_scalar("filling_number");
    // //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // // }

    // // std::cout << meas_handler->BinsNum() << std::endl;
    // // std::cout << meas_handler->BinsSize() << std::endl;

    // std::cout << " wrap error :  " << walker->WrapError() << std::endl;

    // if (meas_handler->find("equaltime_sign")) {
    //     const auto obs = meas_handler->find<Observable::ScalarObs>("equaltime_sign");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }

    // if (meas_handler->find("dynamic_sign")) {
    //     auto obs = meas_handler->find<Observable::ScalarObs>("dynamic_sign");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }
      
    // if (meas_handler->find("filling_number")) {
    //     auto obs = meas_handler->find<Observable::ScalarObs>("filling_number");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }

    // if (meas_handler->find("double_occupancy")) {
    //     auto obs = meas_handler->find<Observable::ScalarObs>("double_occupancy");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }

    // if (meas_handler->find("kinetic_energy")) {
    //     auto obs = meas_handler->find<Observable::ScalarObs>("kinetic_energy");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }

    // if (meas_handler->find("local_spin_corr")) {
    //     auto obs = meas_handler->find<Observable::ScalarObs>("local_spin_corr");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }

    // if (meas_handler->find("momentum_distribution")) {
    //     auto obs = meas_handler->find<Observable::ScalarObs>("momentum_distribution");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }

    // if (meas_handler->find("spin_density_structure_factor")) {
    //     auto obs = meas_handler->find<Observable::ScalarObs>("spin_density_structure_factor");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }

    // if (meas_handler->find("charge_density_structure_factor")) {
    //     auto obs = meas_handler->find<Observable::ScalarObs>("charge_density_structure_factor");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }

    // if (meas_handler->find("s_wave_pairing_corr")) {
    //     auto obs = meas_handler->find<Observable::ScalarObs>("s_wave_pairing_corr");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }

    // // important!!!
    // // todo: the correctness of superfluid stiffness measurements should be further checked in attractive hubbard model
    // if (meas_handler->find("superfluid_stiffness")) {
    //     auto obs = meas_handler->find<Observable::ScalarObs>("superfluid_stiffness");
    //     std::cout << obs.name() << "  " << obs.mean_value() << "  " << obs.error_bar() << std::endl;
    // }

    // // if (meas_handler->find("greens_functions")) {
    // //     auto obs = meas_handler->find<Observable::MatrixObs>("greens_functions");
    // //     std::cout << obs.name() << std::endl;
    // //     for (int t = 0; t < walker->TimeSize(); ++t) {
    // //         std::cout << t << "     " 
    // //                   << obs.mean_value()(1,t) << "      " 
    // //                   << obs.error_bar()(1,t) 
    // //                   << std::endl;
    // //     }
    // // }

    // // if (meas_handler->find("density_of_states")) {
    // //     auto obs = meas_handler->find<Observable::VectorObs>("density_of_states");
    // //     std::cout << obs.name() << std::endl;
    // //     for (int t = 0; t < walker->TimeSize(); ++t) {
    // //         std::cout << t << "     " 
    // //                   << obs.mean_value()(t) << "      " 
    // //                   << obs.error_bar()(t) 
    // //                   << std::endl;
    // //     }
    // // }

    // // std::cout << lattice->Index2Momentum(5) << std::endl;

    // // std::cout << walker->GreenttUp() << std::endl;
    // // std::cout << std::endl;
    // // std::cout << walker->Greent0Up() << std::endl;
    // // std::cout << std::endl;
    // // std::cout << lattice->HoppingMatrix() << std::endl;

    // // std::cout << std::endl;
    // // std::cout << meas_handler->isWarmUp() << std::endl;
    // // std::cout << meas_handler->isEqualTime() << std::endl;
    // // std::cout << meas_handler->isDynamic() << std::endl;
    // // std::cout << meas_handler->find_scalar("filling_number").name() << std::endl;
    // // std::cout << meas_handler->find_matrix("greens_functions").name() << std::endl;















    // // test toml++

    // auto config = toml::parse_file( "../config.toml" );

    // // get key-value pairs
    // const std::string_view model_type = config["Model"]["type"].value_or("RepulsiveHubbard");
    // const auto checker_board = config["CheckerBoard"]["whether_or_not"].value_or(true);

    // // std::cout << model_type << std::endl;
    // // std::cout << checker_board << std::endl;

    // // std::vector<int> lattice_size;
    // // toml::array* arr = config["Lattice"]["cell"].as_array();
    // // if (arr && arr->is_homogeneous<int64_t>()) {
    // //     lattice_size.reserve(arr->size());
    // //     for (auto&& el : *arr) {
    // //         lattice_size.emplace_back(el.value_or(0));
    // //     }
    // // }
    // // std::cout << lattice_size.size() << std::endl;
    // // for (auto el : lattice_size) {
    // //     std::cout << el << std::endl;
    // // }


    // std::vector<std::string_view> observables;
    // toml::array* arr = config["Measure"]["observables"].as_array();
    // if (arr && arr->is_homogeneous<std::string>()) {
    //     observables.reserve(arr->size());
    //     for (auto&& el : *arr) {
    //         observables.emplace_back(el.value_or(""));
    //     }
    // }
    // std::cout << observables.size() << std::endl;
    // for (auto el : observables) {
    //     std::cout << el << std::endl;
    // }
    

    // // if ( toml::array* arr = config["Lattice"]["cell"].as_array() )
    // // {
    // //     // visitation with for_each() helps deal with heterogeneous data
    // //     if (arr && arr->is_homogeneous<int64_t>()) {
            
    // //         arr->for_each([](auto&& el)
    // //         {   
    // //             extern std::vector<int> lattice_size;
    // //             lattice_size.emplace_back(el.value_or(1));

    // //             if constexpr (toml::is_integer<decltype(el)>) {
    // //                 // lattice_size.emplace_back(el);
    // //             }
                
    //     //         // lattice_size.push_back(el);
    //     //         // if (arr && arr->is_homogeneous<int64_t>()) {
                    
    //     //         // }
    //     //     });
    //     // }
        
    // //     std::cout << lattice_size.size() << std::endl;
    // //     std::cout << lattice_size[0] << std::endl;
    // //     std::cout << lattice_size[1] << std::endl;
    // // }

    // // std::cout << lattice_size.size() << std::endl;

    // // auto config = toml::parse_file( "../configuration.toml" );

    // // // // get key-value pairs
    // // // std::string_view library_name = config["library"]["name"].value_or("sv");
    // // // std::string_view library_author = config["library"]["authors"][0].value_or("sv");
    // // // int64_t depends_on_cpp_version = config["dependencies"]["cpp"].value_or(0);

    // // // modify the data
    // // config.insert_or_assign("alternatives", toml::array{
    // //     "cpptoml",
    // //     "toml11",
    // //     "Boost.TOML"
    // // });

    // // // use a visitor to iterate over heterogenous data
    // // config.for_each([](auto& key, auto& value)
    // // {
    // //     std::cout << value << "\n";
    // //     if constexpr (toml::is_string<decltype(value)>) {
    // //         // do_something_with_string_values(value);
    // //     }
    // // });

    // // // // you can also iterate more 'traditionally' using a ranged-for
    // // // for (auto&& [k, v] : config)
    // // // {
    // // //     // ...
    // // // }

    // // // // re-serialize as TOML
    // // // std::cout << config << "\n";

    // // // // re-serialize as JSON
    // // // std::cout << toml::json_formatter{ config } << "\n";

    // // // // re-serialize as YAML
    // // // std::cout << toml::yaml_formatter{ config } << "\n";


    // // if (toml::array* arr = config["Lattice"]["cell"].as_array())
    // // {
    // //     // visitation with for_each() helps deal with heterogeneous data
    // //     arr->for_each([](auto&& el)
    // //     {
    // //         if constexpr (toml::is_number<decltype(el)>)
    // //             (*el)++;
    // //         else if constexpr (toml::is_string<decltype(el)>)
    // //             el = "five"sv;
    // //     });

    // //     // arrays are very similar to std::vector
    // //     arr->push_back(7);
    // //     arr->emplace_back<toml::array>(8, 9);
    // //     std::cout << "numbers: " << numbers << "\n";
    // // }























    // // test lattice momentum
    // int ll = 4;

    // Lattice::LatticeBase* lattice = new Lattice::Square();

    // lattice->set_lattice_params({ll,ll});
    // lattice->initial();

    // std::cout << lattice->Displacement(3,2) << std::endl; 
    // std::cout << lattice->Displacement(2,4) << std::endl; 

    // // std::cout << lattice->Index2Site(1) << std::endl;
    // // std::cout << lattice->Index2Momentum(lattice->GammaPointIndex()) << std::endl;
    // // std::cout << lattice->Index2Momentum(lattice->XPointIndex()) << std::endl;
    // // std::cout << lattice->Index2Momentum(lattice->MPointIndex()) << std::endl;
    
    // // std::cout << lattice->m_index2site_table << std::endl;
    // // std::cout << lattice->m_index2momentum_table << std::endl;

    // // std::cout << lattice->m_fourier_factor_table << std::endl;
    // // std::cout << lattice->FourierFactor(1,1) << std::endl;

    // // for (auto i : lattice->m_k_stars_index) {
    // //     std::cout << i << std::endl;
    // // }

    // // std::cout << lattice->m_gamma_point_index << std::endl;
    // // std::cout << lattice->m_x_point_index << std::endl;
    // // std::cout << lattice->m_m_point_index << std::endl;

    // // for (auto i : lattice->m_delta_line_index ) {
    // //     std::cout << i << std::endl;
    // // }
    // // std::cout << std::endl;

    // // for (auto i : lattice->m_z_line_index ) {
    // //     std::cout << i << std::endl;
    // // }
    // // std::cout << std::endl;

    // // for (auto i : lattice->m_sigma_line_index ) {
    // //     std::cout << i << std::endl;
    // // }
    // // std::cout << std::endl;
    
    // // for (auto i : lattice->m_gamma2x2m2gamma_loop_index ) {
    // //     std::cout << i << std::endl;
    // // }
    // // std::cout << std::endl;
    
















    // // test progress bar
    // const int total = 10000;

    // /*
    //  * Define a progress bar that has a total of 10000,
    //  * a width of 70, shows `#` to indicate completion
    //  * and a dash '-' for incomplete
    //  */
    // progresscpp::ProgressBar progressBar(total, 70, '#', '-');

    // for (int i = 0; i < total; i++) {
        

    //     usleep(200); // simulate work

    //     ++progressBar; // record the tick

    //     // display the bar only at certain steps
    //     // if (i % 10 == 0)
        
    //     std::cout << "hello "; progressBar.display();
    // }

    // // tell the bar to finish
    // std::cout << "hello "; progressBar.done();

    // std::cout << "Done!" << std::endl;




    








    // test lattice


















    // todo: test checkerboard
    // even lattice size (ok) and efficiency (ok)
    
    // todo: trans mult ( finished, trans is actually unnecessary, just use expK )
    // todo: Vmat mult benchmark ( two-times faster! )
    // !!!


    // todo: functional ptr to mult_expK in Model (ok!)
    // ( combine std::function with std::bind to wrap the member function )
    // !!




    // Model::ModelBase* model_cb = new Model::RepulsiveHubbard();
    // Model::ModelBase* model_direct = new Model::RepulsiveHubbard();
    // model_cb->set_model_params(hopping_t, onsite_u, chemical_potential);
    // model_direct->set_model_params(hopping_t, onsite_u, chemical_potential);

    // Lattice::LatticeBase* lattice = new Lattice::Square();
    // QuantumMonteCarlo::DqmcWalker* walker = new QuantumMonteCarlo::DqmcWalker();
    // Measure::MeasureHandler* meas_handler = new Measure::MeasureHandler();
    // CheckerBoard::CheckerBoardBase* checkerboard = new CheckerBoard::Square();

    // lattice->set_lattice_params({ll,ll});
    // walker->set_physical_params(beta, lt);
    // walker->set_stabilization_pace(nwrap);
    // meas_handler->set_measure_params(sweeps_warmup, bin_num, bin_size, sweeps_between_bins);
    // meas_handler->set_observables(obs_list);
    
    // QuantumMonteCarlo::DqmcInitializer::initial_modules(*lattice, *model_cb, *walker, *meas_handler, *checkerboard);
    // QuantumMonteCarlo::DqmcInitializer::initial_modules(*lattice, *model_direct, *walker, *meas_handler);

    // Utils::Random::set_seed_fixed(12345);
    // model_cb->set_bosonic_fields_to_random();
    // Utils::Random::set_seed_fixed(12345);
    // model_direct->set_bosonic_fields_to_random();

    // Eigen::MatrixXd mat_cb = Eigen::MatrixXd::Identity(ll*ll, ll*ll);
    // Eigen::MatrixXd mat_direct = mat_cb;

    // const int num_mult = 1e4;

    // std::chrono::steady_clock::time_point begin_t{}, end_t{};

    // begin_t = std::chrono::steady_clock::now();
    // for (auto i = 0; i < num_mult; ++i) {
    //     model_direct->mult_B_from_left(mat_direct, 0, 1);
    // }
    // end_t = std::chrono::steady_clock::now();
    // std::cout << "direct : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - begin_t).count() << std::endl;
    
    // begin_t = std::chrono::steady_clock::now();
    // for (auto i = 0; i < num_mult; ++i) {
    //     model_cb->mult_B_from_left(mat_cb, 0, 1);
    // }
    // end_t = std::chrono::steady_clock::now();
    // std::cout << "cb : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - begin_t).count() << std::endl;

    // std::cout << (mat_direct - mat_cb).maxCoeff() << std::endl;









    // // test ObservableHandler 
    // Observable::ObservableHandler* handler = new Observable::ObservableHandler();

    // std::vector<std::string> obs_list{ "filling_number", };
    // handler->initial(obs_list);

    // if ( handler->find("filling_number") ) {
    //     std::cout << "found!" << std::endl;
    //     const auto obs = handler->find_scalar("filling_number");
    //     std::cout << obs.name() << std::endl;
    // }
    // if ( handler->find("eqtime_sign") ) {
    //     std::cout << "found!" << std::endl;
    //     const auto obs = handler->find_scalar("eqtime_sign");
    //     std::cout << obs.name() << std::endl;
    // }
    // std::cout << (handler->m_eqtime_scalar_obs[0])->name() << std::endl;
    
    // Measure::MeasureHandler* meas_handler = new Measure::MeasureHandler();
    // Model::ModelBase* model = new Model::ModelBase();
    // Lattice::LatticeBase* lattice = new Lattice::Square2d();

    // (handler->m_eqtime_scalar_obs[0])->measure(*meas_handler, *model, *lattice);








    // // test utils

    // Utils::SvdStack* svd_stack = new Utils::SvdStack(4,10);
    // // Utils::FFTSolver::FFTSolver2d* solver = new Utils::FFTSolver::FFTSolver2d();

    // Eigen::MatrixXd mat = Eigen::MatrixXd::Random(4,4);
    
    // svd_stack->push(mat);
    // svd_stack->push(mat);

    // const auto& u = svd_stack->MatrixU();
    // const auto& s = svd_stack->SingularValues().asDiagonal();
    // const auto& v = svd_stack->MatrixV();
    // std::cout << (u*s*v.transpose() - mat*mat).maxCoeff() << std::endl;

    // Utils::Random::set_seed(1);









    // // test measure

    // Observable::ObservableBase* obs = new Observable::Observable<Observable::ScalarType>();

    // Observable::Observable<Observable::ScalarType>* casted_obs = (dynamic_cast<Observable::Observable<Observable::ScalarType>*>(obs));
    // casted_obs->set_observable_name("filling");
    // std::cout << casted_obs->name() << std::endl;

    // Observable::ObservableHandler* handler = new Observable::ObservableHandler();








    // // test model
    // Model::ModelBase* model = new Model::ModelBase();

    




    // // test lattice

    // Lattice::Square2d mylattice(4);
    // mylattice.initial();

    // std::cout << mylattice.SpaceDim() << std::endl;
    // std::cout << mylattice.SpaceSize() << std::endl;
    // std::cout << mylattice.TotalSiteNum() << std::endl;

    // std::cout << mylattice.site2index({2,3}) << std::endl;
    // auto site = mylattice.index2site(11);
    // for (long unsigned int i = 0; i < site.size(); ++i) {
    //     std::cout << site[i] << std::endl;
    // }

    // std::cout << mylattice.HoppingMatrix() << std::endl;

    // Lattice::LatticeBase* lattice = new Lattice::Square2d(3);
    // lattice->initial();
    // std::cout << lattice->HoppingMatrix() << std::endl;

    return 0;
}
