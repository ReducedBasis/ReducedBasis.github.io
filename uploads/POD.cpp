//! -*- mode: c++; coding: utf-8; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; show-trailing-whitespace: t  -*- vim:fenc=utf-8:ft=cpp:et:sw=4:ts=4:sts=4
//!
//! This file is part of the Feel++ library
//!
//! This library is free software; you can redistribute it and/or
//! modify it under the terms of the GNU Lesser General Public
//! License as published by the Free Software Foundation; either
//! version 2.1 of the License, or (at your option) any later version.
//!
//! This library is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//! Lesser General Public License for more details.
//!
//! You should have received a copy of the GNU Lesser General Public
//! License along with this library; if not, write to the Free Software
//! Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
//!
//! @file
//! @author Christophe Prud'homme <christophe.prudhomme@feelpp.org>
//! @date 14 Jun 2017
//! @copyright 2017 Feel++ Consortium
//!

#include <feel/feelmor/crbplugin.hpp>
#include <heat2d.hpp>



namespace Feel {

Heat2D::super::betaq_type
Heat2D::computeBetaQ( parameter_type const& mu )
{
    for (int k = 0;k<2;++k)
        this->M_betaAq[k] = mu(k);
    for (int k = 0;k<2;++k)
        this->M_betaFq[0][k] = mu(k);
    this->M_betaFq[1][0] = 1.;
    return boost::make_tuple( this->M_betaAq, this->M_betaFq );
}



/// [initmodel]
void
Heat2D::initModel()
{

    CHECK( is_linear && !is_time_dependent ) << "Invalid model is_linear:" << is_linear << " is_time_dependent:" << is_time_dependent << "\n";
    LOG_IF( WARNING, ((Options&NonLinear) == NonLinear) ) << "Invalid model is_linear:" << is_linear << " is_time_dependent:" << is_time_dependent << "\n";
    LOG_IF( WARNING, ((Options&TimeDependent) == TimeDependent) ) << "Invalid model is_linear:" << is_linear << " is_time_dependent:" << is_time_dependent << "\n";
    /*
     * First we create the mesh
     */
    this->setFunctionSpaces( Pch<3>( loadMesh( _mesh=new Mesh<Simplex<2>> ) ) );

    Dmu->setDimension( 2 );
    /// [parameters]
    auto mu_min = Dmu->element();
    mu_min << 1, 1;
    Dmu->setMin( mu_min );
    auto mu_max = Dmu->element();
    mu_max << 20, 20;
    Dmu->setMax( mu_max );
    /// [parameters]

    this->M_betaAq.resize( 2 );
    this->M_betaFq.resize( 2 );
    this->M_betaFq[0].resize( 2 );
    this->M_betaFq[1].resize( 1 );

    auto u = Xh->element();
    auto v = Xh->element();
    auto mesh = Xh->mesh();
    //lhs
    /// [lhs1]
    auto a0 = form2( _trial=Xh, _test=Xh);
    a0 = integrate( markedelements( mesh,"SR" ),  gradt( u )*trans( grad( v ) )  )
        - integrate( markedfaces( mesh, "BR"), (gradt(u)*id(v)+grad(v)*idt(u))*N() + doption("gamma")*idt(u)*id(v) );
    this->addLhs( { a0 , "mu0" } );
    /// [lhs1]

    /// [lhs2]
    auto a1 = form2( _trial=Xh, _test=Xh);
    a1 = integrate( markedelements( mesh,"SL" ),  gradt( u )*trans( grad( v ) )  )
        - integrate( markedfaces( mesh, "BL"), (gradt(u)*id(v)+grad(v)*idt(u))*N() + doption("gamma")*idt(u)*id(v) );
    this->addLhs( { a1 , "mu1" } );
    /// [lhs2]

    //rhs
    /// [rhs1]
    auto f0 = form1( _test=Xh );
    f0 = integrate( markedfaces( mesh,"BR" ), -expr(soption("functions.f"))*(grad(v)*N()+doption("gamma")*id(v)) );
    this->addRhs( { f0, "mu0" } );
    /// [rhs1]

    /// [rhs2]
    auto f1 = form1( _test=Xh );
    f1 = integrate( markedfaces( mesh,"BL" ), -expr(soption("functions.g"))*(grad(v)*N()+doption("gamma")*id(v)) );
    this->addRhs( { f1, "mu1" } );
    /// [rhs2]

    /// [output]
    auto out1 = form1( _test=Xh );
    double meas = integrate(elements(mesh),cst(1.)).evaluate()(0,0);
    out1 = integrate( elements( mesh ), id( u )/cst(meas)) ;
    this->addOutput( { out1, "1" } );
    /// [output]

    // auto out2 = form1( _test=Xh );
    // out2 = integrate( elements( mesh ), idv( u )) / integrate( elements( mesh ), cst(1.)).evaluate()(0,0) ;
    // this->addOutput( { out2, "2" } );

    /// [energy]
    auto energy = form2( _trial=Xh, _test=Xh);
    energy = integrate( markedelements( mesh, "SL" ), gradt( u )*trans( grad( v ) )  )
        - integrate(    markedfaces( mesh, "BR" ), ( (id(v)*gradt(u)+idt(u)*grad(v))*N() + doption("gamma")*idt(u)*id(v) ))
        - integrate(    markedfaces( mesh, "BL" ), ( (id(v)*gradt(u)+idt(u)*grad(v))*N() + doption("gamma")*idt(u)*id(v) ))
        + integrate( markedelements( mesh, "SR" ), gradt( u )*trans( grad( v ) )  );
    this->addEnergyMatrix( energy );
    /// [energy]
}
/// [initmodel]

/// [output]
double
Heat2D::output( int output_index, parameter_type const& mu , element_type& u, bool need_to_solve )
{

    //CHECK( ! need_to_solve ) << "The model need to have the solution to compute the output\n";

    auto mesh = Xh->mesh();
    double output=0;
    // right hand side (compliant)
    if ( output_index == 0 )
    {
        output  = integrate( markedfaces( mesh,"BR" ), -mu(0)*expr(soption("functions.f"))*(gradv(u)*N()+doption("gamma")*idv(u)) ).evaluate()(0,0)
            + integrate( markedfaces( mesh,"BL" ), -mu(1)*expr(soption("functions.g"))*(gradv(u)*N()+doption("gamma")*idv(u)) ).evaluate()(0,0);
    }
    else if ( output_index == 1 )
    {
        output = mean(elements(mesh),idv(u))(0,0);
    }
    // else if ( output_index == 2 )
    // {
    //     output = mean(elements(mesh),idv(u)).evaluat()(0,0);
    // }
    else
        throw std::logic_error( "[Heat2d::output] error with output_index : only 0 or 1 " );
    return output;

}
/// [output]

FEELPP_CRB_PLUGIN(Heat2D, heat2d )

}
